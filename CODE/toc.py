import json
import shutil
import uuid
import os
from typing import List

import pymupdf

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from pydantic import BaseModel, Field


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "DATA"
)

VECTOR_DB_DIR = os.path.join(
    BASE_DIR,
    "VECTOR_DB"
)

COLLECTION_NAME = "first_aid_collection"

OLLAMA_LLM = "llama3.2:latest"

OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large"


class PageTOC(BaseModel):

    title: str = Field(
        description=(
            "Short TOC-style title describing "
            "the target page."
        )
    )

    summary: str = Field(
        description=(
            "Brief factual summary of the "
            "actual target page content."
        )
    )

    main_topic: str = Field(
        description=(
            "Main topic covered by the target page."
        )
    )

    subtopics: List[str] = Field(
        description=(
            "Important topics, procedures, "
            "instructions, concepts, or action "
            "points found on the target page."
        )
    )

    evidence_keywords: List[str] = Field(
        description=(
            "Important keywords and terminology "
            "supported by the target page and "
            "useful for retrieval."
        )
    )

    retrieval_context: str = Field(
        description=(
            "Describe the situations, user needs, "
            "questions, injuries, symptoms, "
            "procedures, or conditions for which "
            "this page would be useful during retrieval."
        )
    )


def generate_document_id():

    return uuid.uuid4().hex[:8]


def create_directories(document_dir):

    directories = [
        os.path.join(
            document_dir,
            "original"
        ),
        os.path.join(
            document_dir,
            "text"
        ),
        os.path.join(
            document_dir,
            "images"
        ),
        os.path.join(
            document_dir,
            "toc"
        )
    ]

    for directory in directories:

        os.makedirs(
            directory,
            exist_ok=True
        )


def extract_page_text(page):

    return page.get_text("text").strip()


def extract_page_images(
    pdf,
    page,
    document_dir,
    document_id,
    page_index
):

    image_dir = os.path.join(
        document_dir,
        "images"
    )

    os.makedirs(
        image_dir,
        exist_ok=True
    )

    page_images = []

    images = page.get_images(
        full=True
    )

    for image_number, image in enumerate(
        images,
        start=1
    ):

        xref = image[0]

        image_info = pdf.extract_image(
            xref
        )

        image_bytes = image_info["image"]

        extension = image_info["ext"]

        if extension != "jpeg":

            continue

        filename = (
            f"{document_id}_"
            f"page_{page_index}_"
            f"image_{image_number}."
            f"{extension}"
        )

        image_path = os.path.join(
            image_dir,
            filename
        )

        with open(
            image_path,
            "wb"
        ) as file:

            file.write(
                image_bytes
            )

        page_images.append(
            os.path.join(
                "images",
                filename
            ).replace(
                "\\",
                "/"
            )
        )

    return page_images


def extract_pdf_data(
    pdf,
    document_id,
    document_dir
):

    pages = {}

    image_data = {}

    total_pages = len(pdf)

    for page_index in range(
        total_pages
    ):

        print(
            f"Processing page "
            f"{page_index}/"
            f"{total_pages - 1}"
        )

        page = pdf[page_index]

        current_text = extract_page_text(
            page
        )

        page_images = extract_page_images(
            pdf=pdf,
            page=page,
            document_dir=document_dir,
            document_id=document_id,
            page_index=page_index
        )

        pages[str(page_index)] = current_text

        if page_images:

            image_data[
                str(page_index)
            ] = page_images

    text_output_path = os.path.join(
        document_dir,
        "text",
        f"{document_id}_pages.json"
    )

    with open(
        text_output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            pages,
            file,
            ensure_ascii=False,
            indent=4
        )

    image_output_path = os.path.join(
        document_dir,
        "images",
        f"{document_id}_images.json"
    )

    with open(
        image_output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            image_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    return pages, image_data


def build_toc_prompt(
    target_page,
    previous_page,
    next_page,
    previous_text,
    target_text,
    next_text
):

    return f"""
Create one retrieval-oriented TOC entry
for the TARGET PAGE of a first-aid PDF.

The TARGET PAGE is the primary source.

Previous and next pages are context only.

Do not attribute information to the target
page unless supported by it.

The TOC will be embedded for semantic retrieval.

Make retrieval_context describe when a user
would need this page, using relevant first-aid
situations, symptoms, injuries, procedures,
and natural user-query terminology.

TARGET PAGE {target_page}:

{target_text}


PREVIOUS PAGE {previous_page}:

{previous_text}


NEXT PAGE {next_page}:

{next_text}


Generate only these fields:

- title: short TOC-style title
- summary: factual summary of the target page
- main_topic: main subject of the target page
- subtopics: important procedures, instructions,
  concepts, or actions
- evidence_keywords: important retrieval terms
- retrieval_context: when and why this page
  would be useful


Rules:

- Use only information supported by the target page.
- Neighboring pages are context only.
- Do not invent information.
- Keep all fields concise but informative.
- If the page is a heading, table, image,
  blank page, or reference page, describe
  its actual content accurately.

Do not generate page_number or offset_pages_used.
Python will generate those values.
"""


def generate_toc(
    pages,
    document_id,
    document_dir,
    model_name=OLLAMA_LLM
):

    llm = ChatOllama(
        model=model_name,
        temperature=0
    )

    structured_llm = llm.with_structured_output(
        PageTOC
    )

    total_pages = len(pages)

    toc_entries = []

    for page_index in range(
        total_pages
    ):

        print(
            f"Generating TOC "
            f"{page_index}/"
            f"{total_pages - 1}"
        )

        if page_index > 0:

            previous_page = page_index - 1

            previous_text = pages[
                str(previous_page)
            ]

        else:

            previous_page = None

            previous_text = ""

        current_text = pages[
            str(page_index)
        ]

        if page_index < total_pages - 1:

            next_page = page_index + 1

            next_text = pages[
                str(next_page)
            ]

        else:

            next_page = None

            next_text = ""

        prompt = build_toc_prompt(
            target_page=page_index,
            previous_page=previous_page,
            next_page=next_page,
            previous_text=previous_text,
            target_text=current_text,
            next_text=next_text
        )

        response = structured_llm.invoke(
            prompt
        )

        offset_pages = []

        if previous_page is not None:

            offset_pages.append(
                previous_page
            )

        if next_page is not None:

            offset_pages.append(
                next_page
            )

        toc_entries.append(
            {
                "page_number": page_index,
                "title": response.title,
                "summary": response.summary,
                "main_topic": response.main_topic,
                "subtopics": response.subtopics,
                "offset_pages_used": offset_pages,
                "evidence_keywords": response.evidence_keywords,
                "retrieval_context": response.retrieval_context
            }
        )

    output_path = os.path.join(
        document_dir,
        "toc",
        f"{document_id}_toc.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            toc_entries,
            file,
            ensure_ascii=False,
            indent=4
        )

    return toc_entries


def create_metadata(
    document_id,
    pdf_path,
    total_pages,
    document_dir
):

    metadata = {
        "document_id": document_id,
        "filename": os.path.basename(pdf_path),
        "total_pages": total_pages,
        "page_indexing": "zero-based",
        "collection_name": COLLECTION_NAME,
        "status": "READY"
    }

    output_path = os.path.join(
        document_dir,
        "metadata.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4
        )


def get_vector_store():

    embedding_model = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embedding_model
    )

    return vector_store


def embed_toc(
    toc_entries,
    document_id
):

    print(
        "\nCreating TOC embeddings..."
    )

    vector_store = get_vector_store()

    documents = []

    ids = []

    for entry in toc_entries:

        page_number = entry[
            "page_number"
        ]

        content = (
            f"Title: {entry['title']}\n"
            f"Summary: {entry['summary']}\n"
            f"Main topic: {entry['main_topic']}\n"
            f"Subtopics: "
            f"{', '.join(entry['subtopics'])}\n"
            f"Evidence keywords: "
            f"{', '.join(entry['evidence_keywords'])}\n"
            f"Retrieval context: "
            f"{entry['retrieval_context']}"
        )

        metadata = {
            "document_id": document_id,
            "page_number": page_number,
            "title": entry["title"],
            "main_topic": entry["main_topic"],
            "source": "toc"
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

        ids.append(
            f"{document_id}_page_{page_number}"
        )

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    print(
        f"Embedded {len(documents)} TOC entries."
    )

    return {
        "collection_name": COLLECTION_NAME,
        "embedded_entries": len(documents)
    }


def process_document(
    pdf_path,
    document_id=None
):

    if document_id is None:

        document_id = generate_document_id()

    document_dir = os.path.join(
        DATA_DIR,
        "documents",
        document_id
    )

    create_directories(
        document_dir
    )

    original_path = os.path.join(
        document_dir,
        "original",
        f"{document_id}_source.pdf"
    )

    shutil.copy2(
        pdf_path,
        original_path
    )

    pdf = pymupdf.open(
        pdf_path
    )

    total_pages = len(pdf)

    print(
        f"Total pages: {total_pages}"
    )

    print(
        "\nExtracting PDF data..."
    )

    pages, image_data = extract_pdf_data(
        pdf=pdf,
        document_id=document_id,
        document_dir=document_dir
    )

    pdf.close()

    print(
        "\nPDF extraction completed."
    )

    print(
        "\nGenerating TOC..."
    )

    toc_entries = generate_toc(
        pages=pages,
        document_id=document_id,
        document_dir=document_dir
    )

    print(
        "\nTOC generation completed."
    )

    print(
        "\nEmbedding TOC..."
    )

    embedding_result = embed_toc(
        toc_entries=toc_entries,
        document_id=document_id
    )

    create_metadata(
        document_id=document_id,
        pdf_path=pdf_path,
        total_pages=total_pages,
        document_dir=document_dir
    )

    return {
        "document_id": document_id,
        "filename": os.path.basename(pdf_path),
        "total_pages": total_pages,
        "toc_entries": len(toc_entries),
        "embedded_entries": embedding_result[
            "embedded_entries"
        ],
        "document_directory": document_dir,
        "vector_database": VECTOR_DB_DIR,
        "collection": COLLECTION_NAME
    }




def get_page_text(
    document_id,
    page_number
):

    text_path = os.path.join(
        DATA_DIR,
        "documents",
        document_id,
        "text",
        f"{document_id}_pages.json"
    )

    if not os.path.exists(
        text_path
    ):

        raise FileNotFoundError(
            f"Text file not found for "
            f"document {document_id}"
        )

    with open(
        text_path,
        "r",
        encoding="utf-8"
    ) as file:

        pages = json.load(
            file
        )

    return pages.get(
        str(page_number),
        ""
    )


def delete_document(
    document_id
):

    vector_store = get_vector_store()

    vector_store.delete(
        where={
            "document_id": document_id
        }
    )

    document_dir = os.path.join(
        DATA_DIR,
        "documents",
        document_id
    )

    if os.path.exists(
        document_dir
    ):

        shutil.rmtree(
            document_dir
        )

    return {
        "document_id": document_id,
        "collection": COLLECTION_NAME,
        "deleted": True
    }
