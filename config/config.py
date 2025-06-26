import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "docs")

# Vectorstore directory
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstores", "final_chroma_db")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# List of Large Language Models
MODEL_GEMINI = "gemini-2.0-flash"
MODEL_DEEPSEEK = "deepseek-r1-distill-llama-70b"

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=  GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        )

