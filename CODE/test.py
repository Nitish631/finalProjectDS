from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
from pydantic import BaseModel, Field
from typing import List
import os
import json
from rich import print


class RelevantPages(BaseModel):
    pages: List[int] = Field(
        description="Relevant printed content page numbers from the candidate TOC sections"
    )

embedding_model = OllamaEmbeddings(
    model="mxbai-embed-large"
)


toc_vectorStore = Chroma(
    persist_directory="VECTOR__DB",
    embedding_function=embedding_model,
    collection_name="toc_first_aid"
)
toc_retriever = toc_vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
SECTION_SELECTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a first-aid retrieval filter.

Select every candidate page that is genuinely relevant to the user's query.
Keep pages that provide directly useful first-aid instructions. Do not select a
page only because it shares a broad word such as skin, blood, or injury with the
query.

Interpret short injury statements as requests for immediate first aid. Prefer
pages that explain what to do now, such as controlling bleeding, applying a
dressing, or immobilising an injury. Exclude descriptive or classification-only
pages unless the user asks about types, classification, causes, or symptoms.

Return the relevant page numbers using the structured output schema. Each value
must be a page number from the candidate entries below. Do not invent page
numbers or return ranges. Only return pages that are directly useful.
"""
        ),
        (
            "human",
            """
USER QUERY:
{query}

CANDIDATE PAGES:
{candidates}

Select only the page numbers from the candidate pages above.
"""
        )
    ]
)
section_selector = ChatOllama(
    model="llama3.2:latest",
    temperature=0
).with_structured_output(schema=RelevantPages)


content_vectorStore = Chroma(
    persist_directory="VECTOR__DB",
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
- Give concise, practical first-aid steps.
- Use only procedures supported by the provided source.
- Do not use outside medical knowledge.
- Do not invent treatments, procedures, warnings, or symptoms.
- Do not mention emergency services unless the user's situation is clearly
  life-threatening or the source specifically indicates that emergency
  assistance is required.
- If emergency assistance is clearly required, use Nepal Ambulance: 102 or Nepal police: 100.
- Otherwise, do not mention 102, 100, or 101.
- Do not add unnecessary warnings or disclaimers.
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


chat_history = []

print("\n\n\n               RAG SYSTEM CREATED \n\n\n")
print("PRESS 0 TO EXIT\n")

while True:
    print("===================================================")
    query = input("YOU: ")
    if query == "0":
        break
    docs = toc_retriever.invoke(query)
    candidates = "\n\n".join(
        f"page_number: {doc.metadata.get('page_number', 'unknown')}\n"
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
    pages = sorted(selector_response.pages)
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
            if page_images:
                image_paths.extend(page_images)
        print("THERE ARE THE DATA FOR THE QUERY")
    else:
        content=""
        print("NO CONTENT FOUND")
    print("-"*40)
    if estimate_tokens(chat_history) > 500:
        chat_history = summarize_history(chat_history)

    final_prompt=QUERY_PROMPT_TEMPLATE.invoke({"content":content,"query":query,"chat_history":chat_history})
    print(final_prompt)
    print("-"*40)
    response=LLM_Model.invoke(final_prompt)
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))
    print(f"AI:\n{response.content}")
    print("-"*40)
    print("===================================================")
