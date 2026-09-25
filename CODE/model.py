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
    temperature=0.5
).with_structured_output(
    RelevantSections
)


answer_llm = ChatOllama(
    model=OLLAMA_LLM,
    temperature=0.75
)


section_selector_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict retrieval section selector.

Select the TOP 6 candidate sections that are most directly
useful for answering the cure of the USER QUERY. Do NOT answer the query.

Evaluate each candidate using:
- title
- main_topic
- content
- subtopics
- evidence_keywords
- retrieval_context

RULES:
- Select only directly relevant candidates.
- Each candidate must independently support the query.
- Do not use page proximity or document similarity as
  evidence of relevance.
- Do not select weakly related candidates.
- Do not select repetitive candidates.
- When candidates contain similar information, select the
  most useful one.
- Prefer different and complementary information.
- Return fewer than 6 if fewer are sufficiently relevant.
- Never return the same section twice.

OUTPUT:
- Return at most 6 sections.
- Return only:
  document_id
  page_number
  retrieval_page_range
- Copy these values EXACTLY from the candidates.
- Never calculate, modify, or invent them.
- Every returned section must exist in the candidates.
- Do not return any other candidate fields.
- Do not explain the selection.

If no candidate is relevant, return:

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
You are a strict first-aid AI assistant.

SOURCE RULES:
- Answer ONLY using information supported by the
  SOURCE PAGES.
- Do not use outside medical knowledge.
- Do not guess, invent, or add unsupported treatments,
  medicines, dosages, procedures, symptoms, or warnings.
- Use CHAT HISTORY only to understand the user's context.
- If the source does not contain enough information,
  respond exactly:

"I don't have enough information about it."

EMERGENCY RULES:
- Follow emergency instructions in the SOURCE PAGES.
- If the source says to contact emergency services,
  clearly tell the user to seek emergency help.
- Do not declare an emergency unless supported by
  the SOURCE PAGES.
- Do not weaken or omit emergency instructions.
- Replace emergency numbers from the source with:

Nepal Ambulance: 102
Nepal Police: 100

- Never provide emergency numbers from another country
  or invent an emergency number.

ANSWER STYLE:
- Give only relevant first-aid information.
- Give practical steps only when supported by the source.
- Keep instructions clear and preserve the source's
  order when steps are given.
- Do not mention the retrieval system or source pages.
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
                        ),
                    "subtopics":
                        toc_item.get(
                            "subtopics",
                            []
                        ),
                    "evidence_keywords":
                        toc_item.get(
                            "evidence_keywords",
                            []
                        ),
                    "retrieval_context":
                        toc_item.get(
                            "retrieval_context",
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

    print("SECTION SELECTOR INPUT:",len(candidates))
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

    print("SECTION SELECTOR OUTPUT:",len(response.sections))
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
def get_pages_from_sections2(
    sections
):

    pages = {}

    for section in sections:

        document_id = section.document_id
        page_number=section.page_number
        

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
def get_image_paths2(
    sections
):

    image_paths: dict[str, dict[int, list[str]]] = {}

    for section in sections:

        document_id = section.document_id
        page_number = section.page_number

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

        paths = image_data.get(
            str(page_number),
            []
        )

        if document_id not in image_paths:
            image_paths[document_id] = {}

        if page_number not in image_paths[document_id]:
            image_paths[document_id][page_number] = []

        for path in paths:

            image_name = os.path.basename(path)

            image_path = (
                f"/images/{document_id}/images/{image_name}"
            )

            if image_path not in image_paths[
                document_id
            ][page_number]:

                image_paths[
                    document_id
                ][page_number].append(
                    image_path
                )

    result = []

    for document_id, pages in image_paths.items():

        for page_number, paths in pages.items():

            for path in paths:

                result.append(path)

    return result


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

    pages = get_pages_from_sections2(
        selected_sections
    )
    print("-"*40)
    print(pages)
    print("-"*40)
    image_paths = get_image_paths2(
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
        "image_paths": image_paths,
        "chat_history":
            updated_history
    }