from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks
)

from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from typing import List

import json

from CODE.toc import (
    DATA_DIR,
    generate_document_id,
    process_document,
    delete_document
)

from CODE.model import first_aid


app = FastAPI(
    title="First Aid Document API",
    version="1.0.0"
)
app.mount(
    "/images",
    StaticFiles(
        directory=Path(DATA_DIR) / "documents"
    ),
    name="images"
)

class ChatMessage(BaseModel):

    role: str
    content: str


class FirstAidRequest(BaseModel):

    query: str

    chat_history: List[ChatMessage] = Field(
        default_factory=list
    )


def read_documents_json() -> list:

    json_file_path = (
        Path(DATA_DIR)
        / "documents.json"
    )

    if (
        json_file_path.exists()
        and json_file_path.stat().st_size > 0
    ):

        try:

            with open(
                json_file_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

                return (
                    data
                    if isinstance(data, list)
                    else []
                )

        except json.JSONDecodeError:

            return []

    return []


def write_documents_json(data: list):

    json_file_path = (
        Path(DATA_DIR)
        / "documents.json"
    )

    with open(
        json_file_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


@app.post("/upload-document")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    document_id = generate_document_id()

    upload_dir = (
        Path(DATA_DIR)
        / "uploads"
    )

    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_pdf = (
        upload_dir
        / f"{document_id}.pdf"
    )

    try:

        with open(
            temporary_pdf,
            "wb"
        ) as output_file:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output_file.write(chunk)

        existing_data = (
            read_documents_json()
        )

        new_entry = {
            "document_id": document_id,
            "filename": file.filename,
            "status": "processing",
            "file_path": str(temporary_pdf)
        }

        existing_data.append(
            new_entry
        )

        write_documents_json(
            existing_data
        )

        background_tasks.add_task(
            process_document,
            pdf_path=temporary_pdf,
            document_id=document_id
        )

        return JSONResponse(
            status_code=202,
            content={
                "success": True,
                "message": (
                    "Document uploaded. "
                    "Processing started in background."
                ),
                "document_id": document_id
            }
        )

    except Exception as e:

        if temporary_pdf.exists():
            temporary_pdf.unlink()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/documents")
def get_documents():

    try:

        documents = read_documents_json()

        return {
            "success": True,
            "documents": documents
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.patch("/documents/{document_id}")
def update_document(
    document_id: str,
    status: str = None,
    filename: str = None
):

    try:

        existing_data = (
            read_documents_json()
        )

        document_found = False

        for entry in existing_data:

            if (
                entry.get("document_id")
                == document_id
            ):

                if status is not None:
                    entry["status"] = status

                if filename is not None:
                    entry["filename"] = filename

                document_found = True
                break

        if not document_found:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Document with ID "
                    f"{document_id} not found "
                    f"in tracking records."
                )
            )

        write_documents_json(
            existing_data
        )

        return {
            "success": True,
            "message": (
                "Document metadata "
                "updated successfully."
            ),
            "document_id": document_id
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.delete("/documents/{document_id}")
def remove_document(
    document_id: str
):

    try:

        result = delete_document(
            document_id
        )

        existing_data = (
            read_documents_json()
        )

        updated_data = [
            doc
            for doc in existing_data
            if doc.get("document_id")
            != document_id
        ]

        write_documents_json(
            updated_data
        )

        return {
            "success": True,
            "message": (
                "Document deleted successfully "
                "from disk and metadata records."
            ),
            **result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/first_aid")
def first_aid_route(
    request: FirstAidRequest
):

    try:

        chat_history = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in request.chat_history
        ]

        return first_aid(
            query=request.query,
            chat_history=chat_history
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# python -m uvicorn CODE.route:app --reload --host 0.0.0.0 --port 8000