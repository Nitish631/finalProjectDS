import json
from pathlib import Path
from CODE.constant import DATA_DIR
import bcrypt
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

ADMIN_FILE = Path(DATA_DIR) / "admin.json"


def read_admins() -> list:

    if (
        ADMIN_FILE.exists()
        and ADMIN_FILE.stat().st_size > 0
    ):

        try:

            with open(
                ADMIN_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

                return (
                    data
                    if isinstance(data, list)
                    else []
                )

        except json.JSONDecodeError:

            return []

    return []


def write_admins(admins: list):

    with open(
        ADMIN_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            admins,
            file,
            indent=4,
            ensure_ascii=False
        )
def authenticate_admin(
    email: str,
    password: str
):

    admins = read_admins()

    admin = next(
        (
            admin
            for admin in admins
            if admin.get("email") == email
        ),
        None
    )

    if not bcrypt.checkpw(
        password.encode("utf-8"),
        admin["password"].encode("utf-8")
    ):
        return None

    return admin