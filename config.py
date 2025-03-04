import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "docs")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstores", "db_chroma_1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

LST_LLMS = ["gemini-1.5-pro", "gemini-1.5-flash", "llama3.1:latest", ]
