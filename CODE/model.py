import json
import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rich import print
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


class RelevantSection(BaseModel):

    document_id: str = Field(
        description="Exact document_id from the candidate."
    )

    page_number: int = Field(
        description="Exact page_number from the candidate."
    )

    retrieval_page_range: List[int] = Field(
        description=(
            "Exact retrieval_page_range from the candidate."
        )
    )


class RelevantSections(BaseModel):

    sections: List[RelevantSection]


embedding_model = OllamaEmbeddings(
    model=OLLAMA_EMBEDDING_MODEL
)


vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_model
)
toc_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)


section_selector = ChatOllama(
    model=OLLAMA_LLM,
    temperature=0
).with_structured_output(
    RelevantSections
)


answer_llm = ChatOllama(
    model=OLLAMA_LLM,
    temperature=0.5
)


section_selector_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict retrieval section selector.

You are NOT answering the user's question.

Your task is to select ONLY the candidate sections
that are directly relevant to the user's query.

Evaluate each candidate independently using ONLY
the information provided in that candidate:

- title
- main_topic
- content

STRICT RULES:

- Select a candidate ONLY if its own content is
  relevant to the user's query.
- Do NOT select a candidate because it is next to,
  before, or after another relevant candidate.
- Do NOT select a candidate based only on its
  page number.
- Do NOT assume that nearby pages contain related
  information.
- Do NOT use information from other candidates to
  make a candidate relevant.
- Do NOT include weakly related or unrelated
  candidates.
- Prefer precision over recall.
- If there is insufficient evidence that a
  candidate is relevant, DO NOT select it.
- A candidate must independently justify its
  inclusion.
- Do NOT answer the user's question.
- Do NOT summarize the candidates.

OUTPUT RULES:

- Return ONLY candidates that are relevant.
- Return the EXACT document_id from the selected
  candidate.
- Return the EXACT page_number from the selected
  candidate.
- Return the EXACT retrieval_page_range from the
  selected candidate.
- Do NOT calculate a page number.
- Do NOT calculate a retrieval_page_range.
- Do NOT copy a page number or retrieval range from
  another candidate.
- Do NOT invent any value.
- Do NOT modify any value.
- Every returned section MUST correspond to an
  actual candidate provided in the input.
- If a candidate is not relevant, its page_number
  MUST NOT appear in the output.

If no candidate is directly relevant, return:

{{
    "sections": []
}}
"""
        ),
        (
            "human",
            """
USER QUERY:

{query}

CANDIDATE SECTIONS:

{candidates}
"""
        )
    ]
)


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a first-aid AI assistant.

Answer the user's question using ONLY
the provided source pages.

Rules:

- Do not use outside knowledge.
- Do not invent information.
- Give practical first-aid instructions
  supported by the source.
- Consider the conversation history.
- If the source does not contain enough
  information, say:

"I don't have enough information about it."

If the source mentions emergency number 995,
replace it with Nepal Ambulance: 102.

For emergencies, use:
Nepal Ambulance: 102
Nepal Police: 100
"""
        ),
        (
            "placeholder",
            "{chat_history}"
        ),
        (
            "human",
            """
SOURCE PAGES:

{content}


USER QUERY:

{query}
"""
        )
    ]
)


def search_toc(query):

    print("TOC RETRIEVING")

    results = toc_retriever.invoke(
        query
    )

    matches = []

    for document in results:

        metadata = document.metadata

        document_id = metadata.get(
            "document_id"
        )

        page_number = metadata.get(
            "page_number"
        )

        toc_path = os.path.join(
            DATA_DIR,
            "documents",
            document_id,
            "toc",
            f"{document_id}_toc.json"
        )

        with open(
            toc_path,
            "r",
            encoding="utf-8"
        ) as file:

            toc_data = json.load(file)

        current_index = None

        for index, toc_item in enumerate(toc_data):

            if (
                toc_item.get("page_number")
                == page_number
            ):

                current_index = index
                break

        if current_index is None:
            continue

        start_index = max(
            0,
            current_index - 1
        )

        end_index = min(
            len(toc_data),
            current_index + 2
        )

        for toc_item in toc_data[
            start_index:end_index
        ]:

            matches.append(
                {
                    "document_id":
                        document_id,

                    "page_number":
                        toc_item.get(
                            "page_number"
                        ),

                    "retrieval_page_range":
                        toc_item.get(
                            "offset_pages_used"
                        ),

                    "title":
                        toc_item.get(
                            "title"
                        ),

                    "main_topic":
                        toc_item.get(
                            "main_topic"
                        ),

                    "content":
                        toc_item.get(
                            "summary",
                            ""
                        )
                }
            )

    unique_matches = {}

    for match in matches:

        key = (
            match["document_id"],
            match["page_number"]
        )

        unique_matches[key] = match

    matches = list(
        unique_matches.values()
    )

    matches.sort(
        key=lambda item: (
            item["document_id"],
            item["page_number"]
        )
    )

    print(matches)
    print("-" * 40)

    return matches


def select_sections(
    query,
    candidates
):

    if not candidates:
        return []

    candidates_text = "\n\n".join(
        str(candidate)
        for candidate in candidates
    )

    print("SECTION SELECTOR INPUT:")
    print(candidates_text)
    print("-" * 40)

    response = section_selector.invoke(
        section_selector_prompt.invoke(
            {
                "query": query,
                "candidates": candidates_text
            }
        )
    )

    print("SECTION SELECTOR OUTPUT:")
    print(response)
    print("-" * 40)

    return response.sections


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

    if not os.path.exists(text_path):

        raise FileNotFoundError(
            f"Text file not found for "
            f"document {document_id}"
        )

    with open(
        text_path,
        "r",
        encoding="utf-8"
    ) as file:

        pages = json.load(file)

    return pages.get(
        str(page_number),
        ""
    )


def get_pages_from_sections(
    sections
):

    pages = {}

    for section in sections:

        document_id = section.document_id

        start_page = section.retrieval_page_range[0]
        end_page = section.retrieval_page_range[1]

        for page_number in range(
            start_page,
            end_page + 1
        ):

            key = (
                document_id,
                page_number
            )

            pages[key] = {
                "document_id": document_id,
                "page_number": page_number
            }

    return list(
        pages.values()
    )

def get_image_paths(
    sections
):

    image_paths = []

    for section in sections:

        document_id = section.document_id

        image_data_path = os.path.join(
            DATA_DIR,
            "documents",
            document_id,
            "images",
            f"{document_id}_images.json"
        )

        with open(
            image_data_path,
            "r",
            encoding="utf-8"
        ) as file:

            image_data = json.load(file)

        for page_number in section.retrieval_page_range:

            paths = image_data.get(
                str(page_number),
                []
            )

            for path in paths:

                image_name = os.path.basename(
                    path
                )

                image_paths.append(
                    f"/images/{document_id}/images/{image_name}"
                )

    return image_paths

def retrieve_page_content(
    pages
):

    content_parts = []

    for page in pages:

        document_id = page[
            "document_id"
        ]

        page_number = page[
            "page_number"
        ]

        text = get_page_text(
            document_id=document_id,
            page_number=page_number
        )

        if not text:
            continue

        content_parts.append(
            f"""
DOCUMENT ID: {document_id}
PAGE: {page_number}

{text}
"""
        )

    return "\n\n".join(
        content_parts
    )


def convert_chat_history(
    chat_history
):

    messages = []

    for message in chat_history:

        if message["role"] == "user":

            messages.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] in [
            "ai",
            "assistant"
        ]:

            messages.append(
                AIMessage(
                    content=message["content"]
                )
            )

    return messages


def generate_answer(
    query,
    content,
    chat_history
):

    history = convert_chat_history(
        chat_history
    )

    prompt = answer_prompt.invoke(
        {
            "query": query,
            "content": content,
            "chat_history": history
        }
    )

    response = answer_llm.invoke(
        prompt
    )

    return response.content


def first_aid(
    query,
    chat_history
):
    print("="*60)
    print("You: ",query)
    candidates = search_toc(
        query=query,
    )

    selected_sections = select_sections(
        query=query,
        candidates=candidates
    )

    pages = get_pages_from_sections(
        selected_sections
    )
    print("-"*40)
    print(pages)
    print("-"*40)
    image_paths = get_image_paths(
        selected_sections
    )
    print(image_paths)
    print("-"*40)
    content = retrieve_page_content(
        pages
    )

    if not content:

        content = (
            "No relevant source content "
            "was found."
        )

    response = generate_answer(
        query=query,
        content=content,
        chat_history=chat_history
    )
    print("AI: ",response)
    print("="*60)

    updated_history = list(
        chat_history
    )

    updated_history.append(
        {
            "role": "user",
            "content": query
        }
    )

    updated_history.append(
        {
            "role": "ai",
            "content": response
        }
    )

    return {
        "response": response,
        "images": image_paths,
        "chat_history":
            updated_history
    }