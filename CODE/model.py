from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List


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
page_extraction_model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
).with_structured_output(schema=PageData)
toc_retriever = toc_vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
TOC_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a First Aid TOC router.

Identify every relevant TOC section described by the user query.

Use the relevant sections' retrieval_page_range values.
Return the union of every page number in those ranges. If a cut may
also involve bleeding, include both the wound and bleeding sections.

Do not use content_page_range.
Do not invent page numbers.
Do not answer the medical question.
Return only the page numbers.
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
             You are a first-aid AI assistant providing first-aid guidance for users in Nepal. 
             Your task is to identify the user's injury and provide appropriate first-aid guidance using ONLY the provided source of knowledge.
               IMPORTANT EMERGENCY NUMBER RULE:
                 The source of knowledge may contain emergency phone numbers from another country.
                   NEVER copy, repeat, or use an emergency phone number from the source of knowledge.
                     For all emergency-service instructions, ALWAYS use the following Nepal emergency numbers:
                       - Ambulance: 102 
                       - Police: 100 
                       - Fire: 101 
                    If the source contains numbers such as 995, 911, 999, 112, 111, or any other emergency number,
                    IGNORE those numbers completely and replace them with the appropriate Nepal emergency number.
                      Follow these rules:
                        1. First identify and confirm the injury or medical situation described by the user.
                        2. Use the provided source of knowledge to determine the appropriate first-aid procedure.
                        3. Give the first-aid procedure as clear, numbered, step-by-step instructions.
                        4. Use the source of knowledge for medical procedures, but NEVER copy foreign emergency-service numbers from the source.
                        5. If emergency medical assistance is required, tell the user to call: Ambulance: 102
                        6. If police assistance is required, tell the user to call: Police: 100 
                        7. If fire and rescue assistance is required, tell the user to call: Fire: 101 
                        8. If the source says to call an emergency number without specifying the type of service, use Ambulance 102. 
                        9. If the situation appears life-threatening, clearly advise the user to seek emergency medical assistance and use Ambulance 102. 
                        10. Do not refer to figures, diagrams, images, or figure numbers, even if they are mentioned in the source. 
                        11. Do not invent medical procedures, facts, or instructions that are not supported by the source. 
                        12. Do not diagnose diseases. 
                        13. If the source does not contain enough information to provide reliable first-aid guidance, clearly state that the available information is insufficient. 
                        14. Keep the response practical, concise, and focused on what the user should do now. 
                        15. Never mention that an emergency number was changed or replaced. Simply provide the correct Nepal emergency number. 
                        """ ) ,
                        ("human","""
SOURCE OF KNOWLEDGE: {content} 

USER QUERY: {query} 
""")
] )

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


    results=content_vectorStore._collection.get(
        where={
            "page":{"$in":pageData.pages}
        },
        include=["documents"]
    )
    content=""
    for document in results["documents"]:
        content+=f"\n{document}"
    # final_prompt=QUERY_PROMPT_TEMPLATE.invoke({"content":content,"query":query})
    # response=LLM_Model.invoke(final_prompt)
    # print(f"AI:\n{response.content}")
    print(content)
    print("===================================================")
