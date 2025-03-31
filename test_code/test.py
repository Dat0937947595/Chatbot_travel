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
import datetime
import time
import dateparser

# Load biến môi trường
load_dotenv()


def get_time_function(query):
    # Xác định ngày hiện tại từ hệ thống
    current_date = datetime.datetime.now()
    current_timezone = time.tzname[0]  # Lấy tên múi giờ hiện tại
    
    # Xử lý truy vấn để hiểu ngữ nghĩa dựa trên ngày hiện tại
    parsed_date = dateparser.parse(query, settings={"RELATIVE_BASE": current_date})
    
    if parsed_date:
        formatted_date = parsed_date.strftime("%Y-%m-%d")
        formatted_time = parsed_date.strftime("%H:%M:%S")
        response = f"Thời gian bạn yêu cầu là {formatted_time} ngày {formatted_date}, múi giờ: {current_timezone}."
    else:
        response = f"Không thể xác định thời gian từ truy vấn '{query}'. Thời gian hiện tại là {current_date.strftime('%H:%M:%S')} ngày {current_date.strftime('%Y-%m-%d')}, múi giờ: {current_timezone}."
    
    return response

print(get_time_function("Cho tôi biết 2 ngày tới là ngày bao nhiêu?"))
