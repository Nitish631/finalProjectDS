import os
import json
import pymupdf
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rich import print

path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ASSETS",
    "firstaid1.pdf"
)

IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ASSETS",
    "page_images"
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

IMAGE_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ASSETS",
    "image_data.json"
)

os.makedirs(IMAGE_DIR, exist_ok=True)

loader = PyMuPDFLoader(path)

docs = loader.load()

docs = docs[4:]

for doc in docs:
    doc.metadata["page"] -= 4
pdf = pymupdf.open(path)
image_data = {}
for doc in docs:
    printed_page = doc.metadata["page"]
    pdf_page_index = printed_page + 4
    page = pdf[pdf_page_index]
    image_paths = []
    images = page.get_images()
    for image_number, image in enumerate(images, start=1):
        xref = image[0]
        image_data_pdf = pdf.extract_image(xref)
        image_bytes = image_data_pdf["image"]
        image_extension = image_data_pdf["ext"]
        if image_extension != "jpeg":
            continue
        image_filename = (
            f"page_{printed_page}_image_{image_number}.{image_extension}"
        )
        image_path = os.path.join(
            IMAGE_DIR,
            image_filename
        )
        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)

        relative_image_path = os.path.relpath(image_path, BASE_DIR)
        image_paths.append(relative_image_path.replace("\\", "/"))
    if image_paths:
        image_data[str(printed_page)] = image_paths


pdf.close()
with open(
    IMAGE_DATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        image_data,
        file,
        indent=4
    )
embedding_model = OllamaEmbeddings(
    model="mxbai-embed-large"
)
content_vectorStore = Chroma(
    collection_name="first_aid",
    embedding_function=embedding_model,
    persist_directory="VECTOR__DB"
)
content_vectorStore.delete_collection()
content_vectorStore = Chroma(
    collection_name="first_aid",
    embedding_function=embedding_model,
    persist_directory="VECTOR__DB"
)
content_vectorStore.add_documents(docs)


print("CONTENT VECTOR STORE CREATED")
print("IMAGE DATA CREATED")