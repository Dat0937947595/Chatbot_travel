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

logger = logging.getLogger("Services")

def weather_info_function(chatbot, query):
    """Hàm tra cứu và sinh phản hồi thời tiết chi tiết cho chatbot du lịch."""
    API_KEY = 'b13f85eb589c453522bb1322a6763a8d'
    BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

    # Prompt trích xuất thông tin từ query
    extract_info_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        Bạn là một trợ lý du lịch thông minh, chuyên nghiệp. Nhiệm vụ:
        1. Trích xuất tên tỉnh/thành phố từ câu hỏi.
        2. Kiểm tra tên tỉnh/thành phố có hợp lệ không. Nếu không sử dụng kiến thức của bạn về tỉnh thành phố việt nam để viết lại tên tỉnh/thành phố.
        3. Tên tỉnh/thành phố được trích xuất được viết dưới dạng viết thường không dấu.
        2. Trích xuất số ngày dự báo yêu cầu (VD: "hôm nay" -> 1, "ngày mai" -> 2, "10 ngày tới" -> 10). Nếu không rõ, mặc định là 1.

        **Câu hỏi người dùng**: "{query}"

        **Định dạng đầu ra (JSON)**:
        - {{"city": "<tên thành phố>", "days": <số ngày>}}
        """
    )

    # Chain trích xuất
    extract_chain = extract_info_prompt | chatbot.llm_gemini | JsonOutputParser()
    try:
        extract_result = extract_chain.invoke({"query": query})
        logger.info(f"Extracted result: {extract_result}")
    except Exception as e:
        logger.error(f"Error extracting info: {str(e)}")
        return "<Ask> Tôi không hiểu yêu cầu của bạn. Vui lòng thử lại."

    city = extract_result.get("city")
    days_requested = extract_result.get("days", 1)
    days_to_fetch = min(days_requested, 5)  # Giới hạn API miễn phí

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
            return "<Ask> Không tìm thấy thông tin thời tiết cho thành phố này."

        # Nhóm dữ liệu theo ngày
        forecast_by_day = {}
        for item in data["list"]:
            date = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d")
            if date not in forecast_by_day:
                forecast_by_day[date] = {
                    "temps": [],
                    "descriptions": [],
                    "wind_speeds": [],
                    "humidities": [],
                    "rain": 0
                }
            forecast_by_day[date]["temps"].append(item["main"]["temp"])
            forecast_by_day[date]["descriptions"].append(item["weather"][0]["description"])
            forecast_by_day[date]["wind_speeds"].append(item["wind"]["speed"])
            forecast_by_day[date]["humidities"].append(item["main"]["humidity"])
            # Tổng lượng mưa trong ngày (nếu có)
            forecast_by_day[date]["rain"] = max(forecast_by_day[date]["rain"], item.get("rain", {}).get("3h", 0))

        # Tạo dữ liệu thời tiết
        weather_summary = []
        for i, (date, info) in enumerate(forecast_by_day.items()):
            if i >= days_to_fetch:
                break
            weather_summary.append({
                "date": date,
                "avg_temp": round(sum(info["temps"]) / len(info["temps"]), 1),
                "description": max(set(info["descriptions"]), key=info["descriptions"].count),
                "wind_speed": round(sum(info["wind_speeds"]) / len(info["wind_speeds"]), 1),
                "humidity": round(sum(info["humidities"]) / len(info["humidities"])),
                "rain": round(info["rain"], 1)
            })

    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return f"<Ask> Lỗi khi tra cứu thời tiết: {str(e)}. Bạn vui lòng đặt câu hỏi càng rõ ràng để tôi giúp bạn trả lời tốt hơn nhé."

    # Prompt chuyên nghiệp để sinh phản hồi
    weather_response_prompt = PromptTemplate(
        input_variables=["query", "weather_data", "days_requested"],
        template="""
        Bạn là một trợ lý du lịch chuyên nghiệp, thân thiện và am hiểu thời tiết. Dựa trên dữ liệu thời tiết từ API và câu hỏi của người dùng, hãy cung cấp phản hồi chi tiết, tự nhiên, hữu ích, phù hợp với mục đích du lịch.

        **Câu hỏi người dùng**: "{query}"
        **Dữ liệu thời tiết**: {weather_data}
        **Số ngày yêu cầu**: {days_requested}

        **Hướng dẫn**:
        - Liệt kê dự báo thời tiết từng ngày (tối đa 5 ngày): bao gồm nhiệt độ trung bình, mô tả thời tiết, tốc độ gió, độ ẩm, và lượng mưa (nếu có).
        - Nếu số ngày yêu cầu > 5, giải thích nhẹ nhàng rằng dữ liệu chỉ có đến 5 ngày và gợi ý kiểm tra sau.
        - Cung cấp gợi ý du lịch dựa trên thời tiết:
          - Mưa > 0mm: Đề xuất mang ô, hoạt động trong nhà.
          - Gió > 7 m/s: Cảnh báo gió mạnh, tránh hoạt động ngoài trời như đi biển.
          - Độ ẩm > 80%: Lưu ý cảm giác oi bức.
          - Nhiệt độ < 20°C hoặc > 35°C: Gợi ý trang phục phù hợp.
        - Giữ giọng điệu thân thiện, ngắn gọn, như một người bạn đồng hành.

        **Ví dụ**:
        - Input: "Thời tiết ở Hà Nội trong 10 ngày tới thế nào?"
          Weather Data: [
            {{"date": "2025-03-20", "avg_temp": 22, "description": "mưa nhỏ", "wind_speed": 5, "humidity": 85, "rain": 2}},
            {{"date": "2025-03-21", "avg_temp": 24, "description": "mây rải rác", "wind_speed": 3, "humidity": 70, "rain": 0}}
          ]
          Output: "Dự báo thời tiết ở Hà Nội trong 5 ngày tới (tôi chỉ có dữ liệu đến đó thôi, bạn có thể hỏi lại sau vài ngày nhé!):\n- 20/03: 22°C, mưa nhỏ, gió 5 m/s, độ ẩm 85%, mưa 2mm – Nhớ mang ô vì trời hơi ẩm ướt!\n- 21/03: 24°C, mây rải rác, gió 3 m/s, độ ẩm 70% – Thời tiết dễ chịu, rất hợp để dạo phố.\nThời tiết tổng thể khá mát mẻ, bạn tha hồ khám phá Hà Nội!"
        """
    )

    # Chain sinh phản hồi
    response_chain = weather_response_prompt | chatbot.llm_gemini | StrOutputParser()
    try:
        response = response_chain.invoke({
            "query": query,
            "weather_data": json.dumps(weather_summary, ensure_ascii=False),
            "days_requested": days_requested
        })
        logger.info(f"Generated response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        # Phản hồi dự phòng
        response = f"Dự báo thời tiết ở {city}:\n"
        for day in weather_summary:
            rain_info = f", mưa {day['rain']}mm" if day['rain'] > 0 else ""
            response += f"- {day['date']}: {day['avg_temp']}°C, {day['description']}, gió {day['wind_speed']} m/s, độ ẩm {day['humidity']}%{rain_info}.\n"
        if days_requested > 5:
            response += "Tôi chỉ có dữ liệu 5 ngày thôi, bạn quay lại hỏi thêm sau nhé!"
        return response

# Test
if __name__ == "__main__":
    from src.chatbot import Chatbot
    chatbot = Chatbot()
    query = "Thời tiết ở Hà Nội trong 10 ngày tới thế nào?"
    print(weather_info_function(chatbot, query))