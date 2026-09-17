import json
from pathlib import Path
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pypdf import PdfReader


class PageTopic(BaseModel):

    page_number: int = Field(
        description="The actual PDF page number of the TARGET page."
    )

    title: str = Field(
        description="A short TOC-style heading for the TARGET page."
    )

    summary: str = Field(
        description="A very brief summary of the TARGET page content (20-30 words)."
    )

    main_topic: str = Field(
        description="The broad subject area covered on the TARGET page."
    )

    subtopics: List[str] = Field(
        description="Important subtopics or action points discussed on the TARGET page."
    )

    offset_pages_used: List[int] = Field(
        description="The neighboring page numbers actually used as context."
    )

    evidence_keywords: List[str] = Field(
        description="Important keywords or terms supported by the TARGET page."
    )


def read_pdf_pages(pdf_path: str | Path) -> List[str]:

    reader = PdfReader(str(pdf_path))

    pages: List[str] = []

    for page in reader.pages:

        text = page.extract_text() or ""

        pages.append(text.strip())

    return pages


def build_window_context(
    all_pages: List[str],
    target_index: int
) -> dict:

    total_pages = len(all_pages)

    previous_index = target_index - 1
    next_index = target_index + 1

    previous_page_number = (
        previous_index + 1
        if previous_index >= 0
        else None
    )

    target_page_number = target_index + 1

    next_page_number = (
        next_index + 1
        if next_index < total_pages
        else None
    )

    return {
        "previous_page_number": previous_page_number,
        "target_page_number": target_page_number,
        "next_page_number": next_page_number,

        "previous_page_text":
            all_pages[previous_index]
            if previous_index >= 0
            else "",

        "target_page_text":
            all_pages[target_index],

        "next_page_text":
            all_pages[next_index]
            if next_index < total_pages
            else "",
    }


def build_prompt_for_window(
    window: dict
) -> str:

    previous_page_number = window["previous_page_number"]
    target_page_number = window["target_page_number"]
    next_page_number = window["next_page_number"]

    previous_page_text = window["previous_page_text"]
    target_page_text = window["target_page_text"]
    next_page_text = window["next_page_text"]

    return f"""
You are creating a page-level table of contents for a first-aid PDF.

You are given THREE consecutive PDF pages:

- Previous page: {previous_page_number}
- TARGET page: {target_page_number}
- Next page: {next_page_number}

IMPORTANT:
You must generate a TOC entry ONLY for the TARGET page.

The previous and next pages are provided ONLY as contextual
information to help understand the TARGET page.

DO NOT generate TOC entries for the previous or next pages.

TARGET PAGE:
{target_page_number}
TARGET PAGE CONTENT:
{target_page_text}




PREVIOUS PAGE:{previous_page_number}
PREVIOUS PAGE CONTEXT
{previous_page_text}



NEXT PAGE :{next_page_number}
NEXT PAGE CONTEXT
{next_page_text}


------------------------------------------------------------
RULES
------------------------------------------------------------

1. Generate exactly ONE TOC entry.

2. The TOC entry MUST be for TARGET page
   {target_page_number}.

3. Do NOT create entries for pages
   {previous_page_number} or {next_page_number}.

4. The TARGET page is the primary source of information.

5. Use the previous and next pages ONLY to understand
   incomplete sentences, continuing procedures, headings,
   or topics that span multiple pages.

6. Do not invent information.

7. The title must be short and TOC-like.

8. The summary must briefly describe the actual content
   of the TARGET page.

9. main_topic must describe the main subject of the
   TARGET page.

10. subtopics should contain important topics,
    procedures, instructions, or action points found
    on the TARGET page.

11. evidence_keywords must contain meaningful terms
    supported by the TARGET page.

12. offset_pages_used must contain ONLY neighboring pages
    that were actually useful for understanding the
    TARGET page.

13. If the TARGET page is mostly a heading, figure,
    table, copyright information, blank space, or other
    non-content material, describe that accurately.

14. The page_number MUST be {target_page_number}.

Return ONLY the structured TOC entry for TARGET page
{target_page_number}.
"""


def generate_toc_for_pdf(
    pdf_path: str | Path,
    model_name: str = "llama3.2:latest",
    window_size: int = 3
) -> List[dict]:

    print("=" * 70)
    print("STARTING PDF TOC GENERATION")
    print("=" * 70)

    if window_size != 3:

        raise ValueError(
            "This program is designed to use exactly "
            "3 pages at a time."
        )

    print("\nReading PDF...")

    all_pages = read_pdf_pages(pdf_path)

    if not all_pages:

        raise ValueError(
            f"No pages were extracted from PDF: {pdf_path}"
        )

    total_pages = len(all_pages)

    llm = ChatOllama(
        model=model_name,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(PageTopic)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful document indexing system. "
                "Create one accurate page-level TOC entry "
                "for the specified TARGET page."
            ),
            (
                "human",
                "{input}"
            ),
        ]
    )

    result_entries: List[dict] = []

    total_middle_pages = max(
        0,
        total_pages - 2
    )

    for target_index in range(
        1,
        total_pages - 1
    ):

        target_page_number = target_index + 1

        print("\n" + "-" * 70)

        print(
            f"PROCESSING TARGET PAGE "
            f"{target_page_number}/{total_pages}"
        )

        print(
            f"Context window: "
            f"{target_page_number - 1}, "
            f"{target_page_number}, "
            f"{target_page_number + 1}"
        )

        print(
            f"TARGET: Page {target_page_number}"
        )

        print("-" * 70)

        window = build_window_context(
            all_pages,
            target_index
        )

        final_prompt = prompt.format(
            input=build_prompt_for_window(window)
        )

        print(
            f"Sending pages "
            f"{target_page_number - 1}, "
            f"{target_page_number}, "
            f"{target_page_number + 1} "
            f"to Llama..."
        )

        response = structured_llm.invoke(
            final_prompt
        )

        print(
            f"✓ Llama generated TOC "
            f"for page {target_page_number}"
        )

        result_entries.append(
            {
                "page_number":
                    response.page_number,

                "title":
                    response.title,

                "summary":
                    response.summary,

                "main_topic":
                    response.main_topic,

                "subtopics":
                    response.subtopics,

                "offset_pages_used":
                    response.offset_pages_used,

                "evidence_keywords":
                    response.evidence_keywords,
            }
        )

        print(
            f"✓ TOC: {response.title}"
        )

        print(
            f"✓ Generated entries: "
            f"{len(result_entries)}/{total_middle_pages}"
        )

    return result_entries


def save_results(
    output_path: str | Path,
    entries: List[dict]
) -> None:

    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            entries,
            handle,
            ensure_ascii=False,
            indent=2
        )

        handle.write("\n")

    print("\n" + "=" * 70)
    print("TOC GENERATION COMPLETED")
    print("=" * 70)

    print(
        f"Total TOC entries: {len(entries)}"
    )

    print(
        f"Saved to: {output_file}"
    )


def main():

    base_dir = Path(__file__).resolve().parent.parent

    pdf_path = (
        base_dir /
        "ASSETS" /
        "firstaid1.pdf"
    )

    output_path = (
        base_dir /
        "ASSETS" /
        "table_of_content1.json"
    )

    generated = generate_toc_for_pdf(
        pdf_path=str(pdf_path),
        model_name="llama3.2:latest",
        window_size=3,
    )

    save_results(
        output_path,
        generated
    )


if __name__ == "__main__":
    main()