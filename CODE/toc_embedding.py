from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TOC_FILE = os.path.join(BASE_DIR, "ASSETS", "table_of_content.json")
VECTOR_DB = os.path.join(BASE_DIR, "VECTOR_DB")

with open(TOC_FILE, "r", encoding="utf-8") as toc_file:
    toc_entries = json.load(toc_file)

if not isinstance(toc_entries, list):
    raise ValueError("table_of_content.json must contain a list of TOC entries.")

documents = []
for entry in toc_entries:
    title = entry.get("title", "Untitled")
    summary = entry.get("summary", "")
    main_topic = entry.get("main_topic", "")
    subtopics = entry.get("subtopics", [])
    evidence_keywords = entry.get("evidence_keywords", [])
    page_number = entry.get("page_number", "")
    offset_pages_used = entry.get("offset_pages_used", 0)
    if isinstance(offset_pages_used, list) and len(offset_pages_used) == 0:
        offset_pages_used = 0

    subtopic_text = "; ".join(str(item) for item in subtopics)
    keyword_text = "; ".join(str(item) for item in evidence_keywords)

    page_content = (
        f"Page number: {page_number}\n"
        f"Offset pages used: {offset_pages_used}\n"
        f"Title: {title}\n"
        f"Main topic: {main_topic}\n"
        f"Summary: {summary}\n"
        f"Subtopics: {subtopic_text}\n"
        f"Evidence keywords: {keyword_text}"
    )

    documents.append(
        Document(
            page_content=page_content,
            metadata={
                "page_number": page_number,
                "offset_pages_used": offset_pages_used ,
                # "title": title,
                # "main_topic": main_topic,
                # "subtopics": subtopics,
                # "evidence_keywords": evidence_keywords,
            },
        )
    )

embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    embedding_function=embedding_model,
    collection_name="toc_first_aid",
    persist_directory=VECTOR_DB,
)
vector_store.delete_collection()

vectorStore = Chroma(
    collection_name="toc_first_aid",
    embedding_function=embedding_model,
    persist_directory=VECTOR_DB
)
vectorStore.add_documents(documents)

print("Collection:", vectorStore._collection.name)
print("Document count:", vectorStore._collection.count())
