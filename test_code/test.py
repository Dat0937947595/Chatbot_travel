import os
import sys
# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import requests
import logging
from datetime import datetime, timedelta
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from src.chatbot import Chatbot


logger = logging.getLogger("Services")

def weather_info_function(chatbot, query):
    """Hàm tra cứu và sinh phản hồi thời tiết chi tiết cho các tỉnh/thành phố ở Việt Nam."""
    API_KEY = 'b13f85eb589c453522bb1322a6763a8d'
    if not API_KEY:
        logger.error("OPENWEATHER_API_KEY không được tìm thấy.")
        return "<Ask> Hệ thống chưa được cấu hình để tra cứu thời tiết. Vui lòng thử lại sau."

    BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

    # Prompt để trích xuất tên thành phố và thời gian
    extract_info_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        Bạn là một trợ lý du lịch thông minh. Nhiệm vụ:
        1. Trích xuất tên tỉnh/thành phố từ câu hỏi.
        2. Trích xuất thời gian và chuyển đổi sang số (VD:
            {{
                "hôm nay": 0,
                "ngày mai": 1,
                "ngày kia": 2,
                "tuần sau": 7
            }}
        ). Nếu không có, mặc định là "hôm nay" -> 0.

        **Câu hỏi**: "{query}"

        **Định dạng đầu ra (JSON)**:
        - Nếu tìm thấy: {{"city": "<tên thành phố>", "time": <thời gian>}}
        - Nếu không rõ: {{"ask": "<câu hỏi làm rõ>"}}
        """
    )

    # Chain để trích xuất thông tin
    extract_chain = extract_info_prompt | chatbot.llm_gemini | JsonOutputParser()
    extract_result = extract_chain.invoke({"query": query})
    logger.info(f"Extracted result: {extract_result}")

    city = extract_result['city']
    days_ahead = extract_result['time']

    # Gọi API OpenWeatherMap
    try:
        params = {
            "q": f"{city},VN",
            "appid": API_KEY,
            "units": "metric",
            "lang": "vi",
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if data.get("cod") != "200":
            logger.warning(f"API error: {data.get('message')}")
            return "<Ask> Không tìm thấy thông tin thời tiết cho thành phố này. Vui lòng kiểm tra lại tên."

        # Lọc dữ liệu cho ngày cụ thể
        target_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        forecast_data = [item for item in data["list"] if datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d") == target_date]

        if not forecast_data:
            return f"Không có dữ liệu thời tiết."

        # Tính trung bình nhiệt độ và chọn mô tả chính
        temps = [item["main"]["temp"] for item in forecast_data]
        descriptions = [item["weather"][0]["description"] for item in forecast_data]
        avg_temp = sum(temps) / len(temps)
        main_description = max(set(descriptions), key=descriptions.count)

        weather_summary = {
            "date": target_date,
            "avg_temp": round(avg_temp, 1),
            "description": main_description
        }

    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return f"<Ask> Lỗi khi tra cứu thời tiết: {str(e)}. Vui lòng thử lại."

    # Prompt để sinh phản hồi chi tiết
    weather_response_prompt = PromptTemplate(
        input_variables=["query", "weather_data"],
        template="""
        Bạn là trợ lý du lịch chuyên nghiệp. Dựa trên dữ liệu thời tiết và câu hỏi, hãy trả lời chi tiết, tự nhiên, hữu ích.

        **Câu hỏi**: "{query}"
        **Dữ liệu thời tiết**: {weather_data}

        **Hướng dẫn**:
        - Trả lời thân thiện, phù hợp với du lịch.
        - Tập trung vào ngày được yêu cầu.
        - Cung cấp thông tin tổng quan nếu không có yêu cầu cụ thể.
        """
    )

    # Chain để sinh phản hồi
    response_chain = weather_response_prompt | chatbot.llm_gemini | StrOutputParser()

    try:
        response = response_chain.invoke({
            "query": query,
            "weather_data": json.dumps(weather_summary, ensure_ascii=False)
        })
        logger.info(f"Generated response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Thời tiết tại {city} vào {days_ahead}: {weather_summary['description']}, nhiệt độ trung bình {weather_summary['avg_temp']}°C."

# Test hàm
if __name__ == "__main__":
    chatbot = Chatbot()
    query = "Thời tiết ở thành phố hồ chí minh ngày mai thế nào?"
    response = weather_info_function(chatbot, query)
    print(response)