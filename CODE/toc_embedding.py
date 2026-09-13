from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import json
from rich import print

with open("ASSETS/table_of_content.json", "r", encoding="utf-8") as file:
    toc = json.load(file)["document"]

documents = []
for chapter in toc["chapters"]:
    for section in chapter["sections"]:
        scope = section["medical_scope"]
        search_terms = list(scope["injury_types"])
        search_terms.extend(scope["body_parts"])

        if "bleeding" in section["title"].lower():
            search_terms.extend([
                "blood coming out",
                "bleeding from a cut",
                "bleeding from a wound",
                "stop bleeding",
                "cut hand or finger",
            ])
        if "wound" in section["title"].lower():
            search_terms.extend([
                "cut hand or finger",
                "open wound",
                "skin cut",
            ])

        text = "\n".join([
            f"Chapter: {chapter['title']}",
            f"Section: {section['title']}",
            f"Topics and symptoms: {', '.join(search_terms)}",
            f"Content page range: {section['content_page_range']}",
            f"Retrieval page range: {section['retrieval_page_range']}",
        ])
        documents.append(Document(
            page_content=text,
            metadata={
                "section_id": section["section_id"],
                "content_page_range": section["content_page_range"],
                "retrieval_page_range": section["retrieval_page_range"],
            },
        ))

embedding_model=OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="toc_first_aid",
    persist_directory="VECTOR__DB",
    embedding_function=embedding_model,
)
vector_store.delete_collection()
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="toc_first_aid",
    persist_directory="VECTOR__DB"
)


print("Collection:", vector_store._collection.name)
print("Document count:", vector_store._collection.count())
