import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "docs")

# Vectorstore directory
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstores", "chroma_db_final_metadata")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# List of Large Language Models
LST_LLMS = ["gemini-1.5-pro", "gemini-1.5-flash", "llama3.1:latest"]
