from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import *
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv  
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

class Model:
    def __init__(self):
        # Khởi tạo LLM Gemini
        self.llm_gemini = ChatGoogleGenerativeAI(
            model=MODEL_GEMINI, 
        )
        
        # # Khởi tạo mô hình embedding
        # self.embedding_model = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        # )
        
        # model embeddings
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def get_llm_gemini(self):
        return self.llm_gemini  

    def get_embedding(self):
        return self.embedding_model
