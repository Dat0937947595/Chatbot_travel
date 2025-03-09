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
from prompts.prompt_template import *

s = "ASK: các địa diểm du lịch nổi tiếng?"
print(s.split(":")[1].strip())