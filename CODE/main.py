from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
from pydantic import BaseModel, Field
from typing import List
import os
import json
from rich import print
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


class RelevantPages(BaseModel):
    pages: List[int] = Field(
        description="Relevant printed content page numbers from the candidate TOC sections"
    )

class ChatMessage(BaseModel):
    role:str
    content:str
class FirstAidRequest(BaseModel):
    query:str
    chat_history:List[ChatMessage]=Field(default_factory=list)

embedding_model = OllamaEmbeddings(
    model="mxbai-embed-large"
)


toc_vectorStore = Chroma(
    persist_directory="VECTOR_DB",
    embedding_function=embedding_model,
    collection_name="toc_first_aid"
)
toc_retriever = toc_vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
SECTION_SELECTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a first-aid retrieval filter.

Analyze the user's query and the candidate pages provided below.
Select every candidate page that is relevant to the user's query based on the information contained in the candidate entries.
Return the relevant page numbers with offset_pages_used using the structured output schema.
The page numbers must come only from the candidate entries and offset_pages_used.
donot miss the offset_pages_used if it is present in the candidate entries.
"""
        ),
        (
            "human",
            """
USER QUERY:
{query}

CANDIDATE PAGES:
{candidates}

Select only the page numbers and offset_pages_used from the candidate pages above.
"""
        )
    ]
)
section_selector = ChatOllama(
    model="llama3.2:latest",
    temperature=0
).with_structured_output(schema=RelevantPages)


content_vectorStore = Chroma(
    persist_directory="VECTOR_DB",
    embedding_function=embedding_model,
    collection_name="first_aid"
)
LLM_Model = ChatOllama(
    model="llama3.2:latest",
    temperature=0.5
)

QUERY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a first-aid AI assistant for users in Nepal. Identify the user's injury
and provide first-aid guidance using ONLY the provided source of knowledge.

RULES:
- Identify the user's injury or situation.
- Search for the action that must take for the user's injury or situation in the provided query.
- Give concise, practical first-aid steps.
- Use only procedures supported by the provided source.
- Do not use outside medical knowledge.
- Do not invent treatments, procedures, warnings, or symptoms.
- If mentioned emergency services to call 995 for SCDF change it to call Nepal Ambulance: 102 
- If emergency assistance is clearly required, use Nepal Ambulance: 102 or Nepal police: 100.
- If the provided source does not contain enough information, clearly say 'i dont have enough information about it'.
"""
        ),
        ("placeholder","{chat_history}"),
        (
            "human",
            """
SOURCE OF KNOWLEDGE:

{content}

USER QUERY:
{query}
"""
        )
    ]
)
IMAGE_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ASSETS",
    "image_data.json"
)
with open(IMAGE_DATA_FILE,"r",encoding="utf-8")as image_json_file:
    image_data=json.load(image_json_file)


def estimate_tokens(messages):
    total = 0
    for msg in messages:
        text = str(msg.content)
        total += max(1, len(text.split()) * 1.3)
    return total


def summarize_history(history):
    if len(history) <= 2:
        return history

    summary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize the earlier conversation into a short memory summary. Keep only the important facts like user condition, user needs, and prior first-aid guidance."
            ),
            (
                "human",
                "CONVERSATION HISTORY:\n{history_text}"
            ),
        ]
    )

    history_text = "\n\n".join(
        f"{type(msg).__name__}: {msg.content}"
        for msg in history
    )
    summary_response = LLM_Model.invoke(
        summary_prompt.invoke({"history_text": history_text})
    )
    summary_text = summary_response.content.strip()

    recent_messages = history[-4:]
    return [
        AIMessage(content=f"Earlier conversation summary: {summary_text}")
    ] + recent_messages


app=FastAPI()
app.mount("/images",
          StaticFiles(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)),"ASSETS/page_images")),
          name="page_images")

print("\n\n\n               RAG SYSTEM CREATED \n\n\n")
print("PRESS 0 TO EXIT\n")

@app.post("/first_aid")
def get_first_aid_response(request:FirstAidRequest):
    query=request.query
    chat_history=[HumanMessage(content=msg.content) if msg.role=="user" else AIMessage(content=msg.content) for msg in request.chat_history]
    print("===================================================")
    print(f"YOU: {query}")
    docs = toc_retriever.invoke(query)
    candidates = "\n\n".join(
        f"page_number: {doc.metadata.get('page_number', 'unknown')}\n"
        f"offset_pages_used: {doc.metadata.get('offset_pages_used', 0)}\n"
        f"title: {doc.metadata.get('title', 'Untitled')}\n"
        f"main_topic: {doc.metadata.get('main_topic', '')}\n"
        f"summary: {doc.metadata.get('summary', '')}\n"
        f"subtopics: {doc.metadata.get('subtopics', [])}\n"
        f"{doc.page_content}"
        for doc in docs
    )
    selector_response = section_selector.invoke(
        SECTION_SELECTOR_PROMPT.invoke({
            "query": query,
            "candidates": candidates,
        })
    )
    pages=set(selector_response.pages)
    pages = sorted(pages)
    if pages:
        print("-"*40)
        print("PAGES: ", pages)
        results = content_vectorStore._collection.get(
            where={
                "page": {"$in": pages}
            },
            include=["documents"]
        )
        content = ""
        for document in results["documents"]:
            content+=document
        image_paths=[]
        for page in pages:
            page_images=image_data.get(str(page),[])
            for image_path in page_images:
                image_name=os.path.basename(image_path)
                image_paths.append(f"/images/{image_name}")
        print("THERE ARE THE DATA FOR THE QUERY")
    else:
        content=""
        print("NO CONTENT FOUND")
    print("-"*40)
    if estimate_tokens(chat_history) > 500:
        chat_history = summarize_history(chat_history)
    print(content)
    print("-"*40)

    final_prompt=QUERY_PROMPT_TEMPLATE.invoke({"content":content,"query":query,"chat_history":chat_history})
    print("-"*40)
    response=LLM_Model.invoke(final_prompt)
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))
    print(f"AI:\n{response.content}")
    print("-"*40)
    print("===================================================")
    return{
        "response":response.content,
        "image_paths":image_paths,
        "chat_history":[
            {"role":"user","content":msg.content} if isinstance(msg,HumanMessage) else {"role":"ai","content":msg.content}
            for msg in chat_history]
    }

# python -m uvicorn CODE.main:app --reload --host 0.0.0.0 --port 8000