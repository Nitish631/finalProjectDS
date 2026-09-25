import os
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