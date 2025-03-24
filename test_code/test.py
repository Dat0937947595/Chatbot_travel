import os
import sys
from functools import partial
from dotenv import load_dotenv
import logging

# Thêm thư mục gốc (CHATBOT_TRAVEL) vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import json
from datetime import datetime, timedelta
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import logging

from src.chatbot import Chatbot

from langchain_community.utilities import OpenWeatherMapAPIWrapper

from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()


print(os.getenv("GOOGLE_API_KEY"))
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash", 
#     google_api_key="AIzaSyA-MAlE62P8Gg2g664zwnYcRAtNykEg_tE"
# )

# from googlesearch import search

# def google_search(query, num_results=5):
#     search_results = search(query, num_results=num_results)
#     return search_results

# # Ví dụ sử dụng
# results = google_search("giá vé xe khách từ TP.HCM đi Đà Lạt")
# for result in results:
#     print(result)
