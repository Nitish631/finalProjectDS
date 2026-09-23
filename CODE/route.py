from pathlib import Path
import random
import smtplib
import os
from datetime import datetime ,timedelta
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks
)
from email.message import EmailMessage
import bcrypt
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from CODE.webiomodels import *
from CODE.r_w_json import *
from dotenv import load_dotenv
load_dotenv(
    Path(__file__).resolve().parent.parent / ".env"
)

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

async def update_delete(
    file,
    document_id,
    adminData: AdminData,
    background_tasks: BackgroundTasks
):

    await upload_document(
        request=adminData,
        background_tasks=background_tasks,
        file=file
    )

    delete_document(
        document_id
    )

@app.post("/upload-document")
async def upload_document(
    request:AdminData,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    admin = authenticate_admin(
        request.email,
        request.password
    )

    if admin is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials"
        )
    
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
def get_documents(request:AdminData):
    admin = authenticate_admin(
        request.email,
        request.password
    )
    if admin is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials"
        )
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
@app.patch("/documents/{document_id}")
async def update_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    adminData: AdminData,
    file: UploadFile = File(...)
):

    admin = authenticate_admin(
        adminData.email,
        adminData.password
    )

    if admin is None:

        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials"
        )

    existing_data = read_documents_json()

    document = next(
        (
            document
            for document in existing_data
            if document.get("document_id")
            == document_id
        ),
        None
    )

    if document is None:

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    await update_delete(
        file=file,
        document_id=document_id,
        adminData=adminData,
        background_tasks=background_tasks
    )

    existing_data = read_documents_json()

    existing_data = [
        document
        for document in existing_data
        if document.get("document_id")
        != document_id
    ]

    write_documents_json(
        existing_data
    )

    return {
        "success": True,
        "message": "Document updated successfully."
    }


@app.delete("/documents/{document_id}")
def remove_document(
    document_id: str,
    request:AdminData
):
    admin = authenticate_admin(
        request.email,
        request.password
    )

    if admin is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin credentials"
        )
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


@app.post("/send-otp")
def send_otp(email_message: EmailRequest):
    email=email_message.email
    admins = read_admins()

    admin = next(
        (
            admin
            for admin in admins
            if admin.get("email") == email
        ),
        None
    )

    if admin is None:
        return {
            "message": "Admin email not found"
        }

    otp = str(
        random.randint(100000, 999999)
    )

    otp_expiry = (
        datetime.now()
        + timedelta(minutes=5)
    )

    admin["otp"] = otp
    admin["otp_expiry"] = (
        otp_expiry.isoformat()
    )
    admin["otp_verified"] = False

    write_admins(admins)

    message = EmailMessage()
    message["Subject"] = "Your OTP"
    message["From"] = os.getenv(
        "EMAIL_ADDRESS"
    )
    message["To"] = email

    message.set_content(
        f"Your OTP is: {otp}\n\n"
        "This OTP will expire in 5 minutes."
    )

    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465
    ) as server:

        server.login(
            os.getenv("EMAIL_ADDRESS"),
            os.getenv("EMAIL_APP_PASSWORD")
        )

        server.send_message(message)

    return {
        "message": "OTP sent successfully"
    }


@app.post("/verify_otp")
def verify_otp(
    request: VerifyOtpRequest
):

    admins = read_admins()

    admin = next(
        (
            admin
            for admin in admins
            if admin.get("email")
            == request.email
        ),
        None
    )

    if admin is None:
        return {
            "message": "Admin email not found"
        }

    if (
        "otp" not in admin
        or "otp_expiry" not in admin
    ):
        return {
            "message": "OTP not generated"
        }

    if admin["otp"] != request.otp:
        return {
            "message": "Invalid OTP"
        }

    otp_expiry = datetime.fromisoformat(
        admin["otp_expiry"]
    )

    if datetime.now() > otp_expiry:
        return {
            "message": "OTP expired"
        }

    admin["otp_verified"] = True

    admin["password_change_time"] = (
        datetime.now()
        + timedelta(minutes=5)
    ).isoformat()

    write_admins(admins)

    return {
        "message": "OTP verified successfully"
    }


@app.post("/change-password")
def change_password(
    request: ChangePasswordRequest
):

    admins = read_admins()

    admin = next(
        (
            admin
            for admin in admins
            if admin.get("email")
            == request.email
        ),
        None
    )

    if admin is None:
        return {
            "message": "Admin email not found"
        }

    if not admin.get(
        "otp_verified",
        False
    ):
        return {
            "message": "OTP not verified"
        }

    if "otp" not in admin:
        return {
            "message": "OTP not found"
        }

    if admin["otp"] != request.otp:
        return {
            "message": "Invalid OTP"
        }

    if "password_change_time" not in admin:
        return {
            "message": (
                "Password change time not found"
            )
        }

    password_change_time = (
        datetime.fromisoformat(
            admin["password_change_time"]
        )
    )

    if datetime.now() > password_change_time:
        return {
            "message": (
                "Password change time expired"
            )
        }

    hashed_password = bcrypt.hashpw(
    request.password.encode("utf-8"),
    bcrypt.gensalt()
    ).decode("utf-8")

    admin["password"] = hashed_password

    admin.pop("otp", None)
    admin.pop("otp_expiry", None)
    admin.pop(
        "password_change_time",
        None
    )

    admin["otp_verified"] = False

    write_admins(admins)

    return {
        "message": (
            "Password changed successfully"
        )
    }

# python -m uvicorn CODE.route:app --reload --host 0.0.0.0 --port 8000