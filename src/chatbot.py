import os
import sys
from functools import partial
from dotenv import load_dotenv
import logging

# Thêm thư mục gốc (CHATBOT_TRAVEL) vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import từ LangChain và các thư viện khác
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Import nội bộ project
from config.config import VECTORSTORE_DIR  # Giả sử bạn đã định nghĩa các hằng số trong config
from src.services import *

# Import các prompt template
from prompts.main_prompt_template import main_prompt_template
from prompts.query_generation_prompt_template import query_generation_prompt_template

from src.model import Model

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Chatbot")

# Load biến môi trường
load_dotenv()

class Chatbot:
    def __init__(self, verbose=False):
        """Khởi tạo chatbot với các thành phần cần thiết."""
        # Khởi tạo model và các thành phần LLM, embedding
        self.model = Model()
        self.llm_gemini = self.model.get_llm_gemini()
        self.embedding_model = self.model.get_embedding()
        
        # Lịch sử hội thoại và truy vấn
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Khởi tạo vectorstore
        self.vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=self.embedding_model
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )

        # Chain để tinh chỉnh truy vấn
        self.query_generation_chain = query_generation_prompt_template | self.llm_gemini | JsonOutputParser()

        # Khởi tạo tools và agent
        self.tools = self._initialize_tools()
        self.agent = self._initialize_agent(verbose=verbose)
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=verbose,  #  Tùy chọn bật/tắt log chi tiết
            handle_parsing_errors=True  # Tự động xử lý lỗi parsing
        )

    def _initialize_tools(self):
        return [       
            Tool(
                name="LocalAgent", 
                func=partial(location_info_function, self), 
                description="Thông tin địa điểm từ cơ sở dữ liệu."
                ),
            
            Tool(
                name="WeatherAgent", 
                func=partial(agent_weather, self), 
                description="Thông tin liên quan đến thời tiết."
                ),
            
            Tool(
                name="SearchAgent", 
                func=partial(search_agent, self), 
                description="Tìm kiếm các thông tin từ trang web."),
            
            # Tool(
            #     name="PlanAgent", 
            #     func=partial(itinerary_planner_function, self), 
            #     description="Lập kế hoạch chuyến đi từ cơ sở dữ liệu."
            #     ),
            
        ]

    def _initialize_agent(self, verbose=False):
        """Khởi tạo React Agent."""
        return create_react_agent(
            llm=self.llm_gemini,
            tools=self.tools,
            prompt=main_prompt_template
        )
        
    def _return_history(self):
        """Trả về lịch sử hội thoại."""
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        return chat_history

    def chat(self, user_input):
        """Xử lý đầu vào người dùng và trả về phản hồi."""
        try:
            logger.info(f"Processing user input: {user_input}")
            logger.info(f"History: {self._return_history()}")
            
            # Viết lại truy vấn nếu cần
            rewrite = rewrite_query(self.llm_gemini, user_input, self._return_history())
            logger.info(f"Rewritten query: {rewrite}")
            
            # Kiểm tra tính liên quan của câu hỏi
            relevantchecker = relevant_travel(self.llm_gemini, rewrite['refined_query'])
            logger.info(f"Relevant checker is_relevant: {relevantchecker.is_relevant}")
            logger.info(f"Relevant checker reason: {relevantchecker.reason}")
            logger.info(f"Relevant checker response: {relevantchecker.response}")
            
            # Nếu câu hỏi không liên quan, trả về phản hồi
            if not relevantchecker.is_relevant:
                # Nếu câu hỏi không liên quan, trả về phản hồi
                return relevantchecker.response

            # Nếu câu hỏi liên quan, kiểm tra lại truy vấn
            query_checker = validate_query(self.llm_gemini, rewrite['refined_query'], self._return_history())
            logger.info(f"Query checker result: {query_checker['result']}")
            
            # Nếu cần hỏi người dùng, lưu lại ngữ cảnh
            if query_checker['ask_human']:
                self.memory.save_context(
                    {"inputs": rewrite['refined_query']},
                    {"outputs": query_checker['result']}
                )
                return query_checker['result']

            # Gọi agent để xử lý truy vấn
            response = self.executor.invoke({"input": user_input})
            output_text = response.get("output", "Không có phản hồi từ agent.")

            # logger.info(f"Generated response: {output_text}")
            return output_text

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Đã xảy ra lỗi: {str(e)}"

    def reset_memory(self):
        """Xóa bộ nhớ hội thoại."""
        self.memory.clear()
        logger.info("Conversation memory has been reset.")