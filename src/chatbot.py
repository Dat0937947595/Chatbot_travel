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
from functools import partial
from dotenv import load_dotenv
import logging

# Import nội bộ project
from config.config import *
from src.services import *
from prompts.prompt_template import *
from src.model import *
# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Query Processing")

# Cấu hình môi trường
# dotenv.load_dotenv("api.env")
load_dotenv()

class Chatbot:
    def __init__(self):
        self.model = Model()
        # Khởi tạo LLM và embedding
        self.llm_gemini = self.model.get_llm_gemini()
        self.embedding_model = self.model.get_embedding()

        # Biến lưu trữ lịch sử hội thoại và truy vấn
        self.history_conversation = []
        self.query = ""
        
        # Khởi tạo vectorstore
        self.vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=self.embedding_model
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.query_generation_chain = LLMChain(
            llm=self.llm_gemini,
            prompt=query_generation_prompt_template
        )
        # Định nghĩa các tool==================
        # Tool tìm kiếm thông tin từ lịch sử
        self.tool_memory_agent = Tool(
            name="MemoryAgent",
            func=partial(query_history, self),
            description="Tìm kiếm thông tin trong lịch sử trò chuyện."
        )

        # Tool tinh chỉnh truy vấn du lịch
        self.tool_travel_agent = Tool(
            name="TravelAgent",
            func=partial(refine_query, self),
            description="Tìm kiếm những thông tin liên quan đến du lịch."
        )

        # Tool cung cấp thông tin địa điểm du lịch
        self.tool_location_info_agent = Tool(
            name="LocationAgent",
            func=partial(location_info_function, self),
            description="Cung cấp thông tin về địa điểm du lịch, bao gồm mô tả, lịch sử, điểm tham quan nổi bật và cách di chuyển."
        )

        # Tool lập kế hoạch du lịch
        self.tool_itinerary_planner_agent = Tool(
            name="PlanAgent",
            func=partial(itinerary_planner_function, self),  # Dùng hàm riêng để lên lịch trình
            description="Hỗ trợ lên lịch trình du lịch chi tiết theo sở thích, thời gian và địa điểm người dùng cung cấp."
        )

        # Tool cung cấp thông tin thời tiết
        self.tool_weather_info_agent = Tool(
            name="WeatherAgent",
            func=partial(weather_info_function, self),  # Dùng hàm riêng lấy dữ liệu thời tiết
            description="Cung cấp thông tin thời tiết tại các địa điểm du lịch, bao gồm nhiệt độ, độ ẩm và dự báo thời tiết."
        )

        # Tool giải đáp câu hỏi về du lịch
        self.tool_travel_faq_agent = Tool(
            name="TravelFAQAgent",
            func=partial(travel_faq_function, self),  # Dùng hàm riêng để trả lời FAQ
            description="Giải đáp các câu hỏi thường gặp về du lịch, kinh nghiệm và mẹo giúp chuyến đi suôn sẻ hơn."
        )


        # Khởi tạo React Agent
        self.react_agent = create_react_agent(
            llm=self.llm_gemini,
            tools=[
                self.tool_memory_agent, 
                self.tool_travel_agent
                ],
            prompt=main_prompt_template
        )

        # Tạo AgentExecutor
        self.agent_search_executor = AgentExecutor(
            agent=self.react_agent,
            tools=[
                self.tool_memory_agent, 
                self.tool_travel_agent
                ],
            memory= self.memory,
            verbose=True,
            handle_parsing_errors=False
        )

        # Khởi tạo React Agent
        self.react_answer_agent = create_react_agent(
            llm=self.llm_gemini,
            tools=[
                self.tool_location_info_agent, 
                self.tool_itinerary_planner_agent, 
                self.tool_weather_info_agent, 
                self.tool_travel_faq_agent
                ],
            prompt=answer_main_prompt_template 
        )

        # Tạo AgentExecutor nhớ tạo sau mỗi bước dùng tool thì làm cách nào để để reset bộ nhớ lại
        self.agent_answer_travel_executor = AgentExecutor(
            agent=self.react_answer_agent,
            tools=[
                self.tool_location_info_agent, 
                self.tool_itinerary_planner_agent, 
                self.tool_weather_info_agent, 
                self.tool_travel_faq_agent],
            memory= self.memory,
            verbose=True,
            handle_parsing_errors=False
        )

    def get_query(self, query):
        """Lưu trữ truy vấn người dùng."""
        self.query = query
    
    def chat(self, user_input):
        """Xử lý đầu vào người dùng và trả về phản hồi."""
        self.get_query(user_input)
        
        self.history_conversation.append({"role": "user", "content": user_input})
        response = self.agent_search_executor.invoke({"input": user_input})
        
        output_text = response.get("output", "")
        self.history_conversation.append({"role": "assistant", "content": output_text})
        
        return output_text