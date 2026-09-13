from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List
import os
import json
from rich import print


class PageData(BaseModel):
    pages: List[int]=[]

    def __str__(self):
        if self.pages:
            return ",".join(map(str,self.pages))
        return ""

    def __repr__(self):
        return self.__str__()

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
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
page_extraction_model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
).with_structured_output(schema=PageData)
TOC_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a First Aid TOC router.

Identify the injury type described in the user query and select the
most relevant TOC section.

Use ONLY the selected section's content_page_range.

Convert content_page_range [start, end] into every page number
from start through end.

Do not use retrieval_page_range.
Do not use any other page range.
Do not invent page numbers.
Do not answer the medical question.

Return only the page numbers from content_page_range.
"""
        ),
        (
            "human",
            """
TOC:

{TOC}

USER QUERY:

{query}
"""
        )
    ]
)


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

print("\n\n\n               RAG SYSTEM CREATED \n\n\n")
print("PRESS 0 TO EXIT\n")

while True:
    print("===================================================")
    query = input("YOU: ")
    if query == "0":
        break
    docs = toc_retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt = TOC_PROMPT_TEMPLATE.invoke({
        "TOC": context,
        "query": query
    })
    pageData = page_extraction_model.invoke(final_prompt)
    if  pageData.pages:
        min_page=min(pageData.pages)
        max_page=max(pageData.pages)
        pageData.pages=list(range(min_page,max_page+1))

        print("-"*40)
        print("PAGES: ",pageData.pages)
        results = content_vectorStore._collection.get(
            where={
                "page": {"$in": pageData.pages}
            },
            include=["documents"]
        )
        content = ""
        for document in results["documents"]:
            content+=document
        image_paths=[]
        for page in pageData.pages:
            page_images=image_data.get(str(page),[])
            if page_images:
                image_paths.extend(page_images)
        print("THERE ARE THE DATA FOR THE QUERY")
    else:
        content=""
        print("NO CONTENT FOUND")
    print("-"*40)
    final_prompt=QUERY_PROMPT_TEMPLATE.invoke({"content":content,"query":query})
    print(final_prompt)
    print("-"*40)
    response=LLM_Model.invoke(final_prompt)
    print(f"AI:\n{response.content}")
    print("-"*40)
    print("IMAGE SOURCE: ")
    print(image_paths)
    print("-"*40)
    print("===================================================")
