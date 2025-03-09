import os
import sys

# Thêm thư mục gốc (CHATBOT_TRAVEL) vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import bên ngoài (package, lib)
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import logging

# Import nội bộ project
from config.config import *
from src.services import *
from prompts.prompt_template import *

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Query Processing")

# Cấu hình môi trường
# dotenv.load_dotenv("api.env")
load_dotenv()

class Chatbot:
    def __init__(self):
        # Khởi tạo LLM
        self.llm_gemini = ChatGoogleGenerativeAI(model=MODEL_GEMINI)
        
        # Sử dụng deep seek
        # self.llm = ChatGroq(
        #     model_name="deepseek-r1-distill-llama-70b",
        #     api_key="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"
        # )
        
        # Biến lưu trữ lịch sử hội thoại và truy vấn
        self.history_conversation = []
        self.query = ""
        
        # Khởi tạo embedding và vectorstore
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=self.embedding_model
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )
        
        # Khởi tạo memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Chuỗi tạo câu hỏi biến thể
        self.query_generation_chain = LLMChain(
            llm=self.llm_gemini,
            prompt=query_generation_prompt_template
        )
        
        # Định nghĩa các tool
        self.tool_memory_agent = Tool(
            name="MemoryAgent",
            func=query_history,
            description="Tìm kiếm thông tin trong lịch sử trò chuyện."
        )
        self.tool_travel_agent = Tool(
            name="TravelAgent",
            func=refine_query,
            description="Tìm kiếm khách sạn dựa trên địa điểm và ngày."
        )
        
        # Khởi tạo React Agent và AgentExecutor
        self.react_agent = create_react_agent(
            llm=self.llm_gemini,
            tools=[self.tool_memory_agent, self.tool_travel_agent],
            prompt=main_prompt_template
        )
        self.agent_search_executor = AgentExecutor(
            agent=self.react_agent,
            tools=[self.tool_memory_agent, self.tool_travel_agent],
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=False
        )
    
    def get_query(self, query):
        """Lưu trữ truy vấn người dùng."""
        self.query = query
    
    def chat(self, user_input):
        """Xử lý đầu vào người dùng và trả về phản hồi."""
        self.get_query(user_input)
        response = refine_query(self, user_input)
        self.history_conversation.append({"role": "assistant", "content": response})
        return response