from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import json
from rich import print

loader=JSONLoader(
    file_path="ASSETS/table_of_content.json",
    jq_schema=".document.chapters[].sections[]",
    text_content=False
)

documents = loader.load()
for doc in documents:
    data=json.loads(doc.page_content)
    print(data["retrieval_page_range"])

embedding_model=OllamaEmbeddings(model="nomic-embed-text")
vectorStrore=Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="toc_first_aid",
    persist_directory="VECTOR__DB"
)
vectorStrore.add_documents(documents)


print("Collection:", vectorStrore._collection.name)
print("Document count:", vectorStrore._collection.count())
