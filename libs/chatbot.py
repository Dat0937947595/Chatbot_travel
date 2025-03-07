from langchain_groq import ChatGroq
import logging
from libs.prompt_template import *
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os


### Cấu hình
dotenv.load_dotenv("api.env")
api_key_llm = os.getenv("GROQ_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Question Processing")

# Đối tượng chatbot
class Chatbot:
    def __init__(self):
        self.llm = ChatGroq(
            model_name="deepseek-r1-distill-qwen-32b",
            temperature=0.7,
            api_key=api_key_llm
        )
        self.memory = ""
        self.query = ""
    
    def get_query(self, query):
        self.query = query
        

    

    def refine_query(self, user_query, chat_history):
        """
        Refine câu hỏi của người dùng dựa trên lịch sử hội thoại, sử dụng prompt mới với phương pháp Chain of Thought (CoT).
        """
        try:
            logger.info(f"Query: {user_query}")
            
            # Thực hiện refine câu hỏi
            refined_query_chain = (
                query_refinement_prompt_template
                | self.llm  # Sử dụng Large Language Model để sinh output
                | StrOutputParser()  # Parser để chuyển đổi output về dạng string
            )
            
            print(chat_history)
            refined_query = refined_query_chain.invoke({"chat_history": chat_history, "user_query": user_query})
        except Exception as e:
            logger.error(f"Lỗi khi refine query: {e}")
            # Trường hợp lỗi, giữ nguyên câu hỏi gốc
            refined_query = user_query
        
        return refined_query