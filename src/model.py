from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import *
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv  

load_dotenv()

class Model:
    def __init__(self):
        # Khởi tạo LLM Gemini
        self.llm_gemini = ChatGoogleGenerativeAI(
            model=MODEL_GEMINI, 
            google_api_key="AIzaSyA-MAlE62P8Gg2g664zwnYcRAtNykEg_tE"
        )
        
        # # Khởi tạo DeepSeek (có thể dùng thay thế Gemini nếu cần)
        # self.llm_deepseek = ChatGroq(
        #     model_name="deepseek-r1-distill-llama-70b",
        #     api_key="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"
        # )
        
        # Khởi tạo mô hình embedding
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # self.vectorstore = Chroma(
        #     persist_directory=VECTORSTORE_DIR,
        #     embedding_function=self.embedding_model
        # )

        # self.retriever = self.vectorstore.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={"k": 5}
        # )

    def get_llm_gemini(self):
        return self.llm_gemini  
    
    # def get_llm_deepseek(self):
    #     return self.llm_deepseek 
    

    def get_embedding(self):
        return self.embedding_model
