"""
    File này dùng để test code
"""

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

# Import nội bộ project
from config.config import VECTORSTORE_DIR
from src.services import *
from src.model import Model
# from prompts.query_generation_prompt_template import query_generation_prompt_template
# from prompts.prompt_template import query_generation_prompt_template
from langchain_core.output_parsers import JsonOutputParser

from src.services import *
from src.chatbot import Chatbot

import json
# from prompts.location_info_prompt_template import location_info_prompt

# chatbot = Chatbot(verbose=True)

# # Lấy danh sách câu hỏi từ query_generation_chain
# def get_questions(x):
#     result = chatbot.query_generation_chain.invoke({"question": x})
#     try:
        
#         logger.info(f"\nKết quả từ query_generation_chain: {result}\n")
#         # Lấy danh sách câu hỏi từ key "questions"
#         return result["questions"]
#     except (KeyError, TypeError) as e:
#         logger.error(f"Lỗi khi xử lý output từ query_generation_chain: {e}")
#         return [x]  # Trả về câu hỏi gốc nếu có lỗi

# cleaned_query_generation = RunnableLambda(get_questions)

# retrieval_chain_rag_fusion = (
#     cleaned_query_generation
#     | chatbot.retriever.map()  # Áp dụng retriever cho từng câu hỏi trong danh sách
#     | RunnableLambda(lambda results: reciprocal_rank_fusion(results))  # Gộp kết quả
# )

# def format_input(inputs):
#     context_result = retrieval_chain_rag_fusion.invoke(inputs)
#     return {"retrieved_context": context_result, "question": inputs["question"]}

# formatted_prompt = RunnableLambda(lambda x: location_info_prompt.format(**x))

# final_rag_chain = (
#     RunnableLambda(format_input)
#     | formatted_prompt
#     | chatbot.llm_gemini
#     | StrOutputParser()
# )

# user_input = "Các địa điểm du lịch nổi tiếng ở Đà Nẵng?"
# print(final_rag_chain.invoke({"question": user_input}))



from prompts.main_prompt_template import main_prompt_template

print(main_prompt_template)