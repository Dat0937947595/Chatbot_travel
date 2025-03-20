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
<<<<<<< Updated upstream
# from prompts.location_info_prompt_template import location_info_prompt

# chatbot = Chatbot(verbose=True)
=======
from datetime import datetime, timedelta
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import logging
logger = logging.getLogger("Services")
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
# cleaned_query_generation = RunnableLambda(get_questions)
=======
    # Ngày hiện tại động
    CURRENT_DATE = datetime.now()

    # Prompt trích xuất thông tin từ query
    extract_info_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        Bạn là một trợ lý du lịch thông minh, chuyên nghiệp. Nhiệm vụ:
        1. Trích xuất tên tỉnh/thành phố từ câu hỏi.
        2. Trích xuất số ngày dự báo yêu cầu (VD: "hôm nay" -> 1, "trong 2 ngày tới" -> 2).
           - Nếu có cụm "trong X ngày tới" hoặc "tới", dự báo bắt đầu từ ngày mai.
           - Nếu chỉ nói "hôm nay" hoặc không rõ, dự báo bắt đầu từ hôm nay.
        3. Xác định xem dự báo có bắt đầu từ ngày mai không.
>>>>>>> Stashed changes

# retrieval_chain_rag_fusion = (
#     cleaned_query_generation
#     | chatbot.retriever.map()  # Áp dụng retriever cho từng câu hỏi trong danh sách
#     | RunnableLambda(lambda results: reciprocal_rank_fusion(results))  # Gộp kết quả
# )

<<<<<<< Updated upstream
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
=======
        **Định dạng đầu ra (JSON)**:
        - {{"city": "<tên thành phố>", "days": <số ngày>, "start_from_tomorrow": <true/false>}}
        """
    )

    # Chain trích xuất
    extract_chain = extract_info_prompt | chatbot.llm_gemini | JsonOutputParser()
    try:
        extract_result = extract_chain.invoke({"query": query})
        logger.info(f"Extracted result: {extract_result}")
    except Exception as e:
        logger.error(f"Error extracting info: {str(e)}")
        return "<Ask> Tôi không hiểu yêu cầu của bạn. Vui lòng thử lại với câu hỏi rõ hơn."

    city = extract_result.get("city")
    days_requested = extract_result.get("days", 1)
    start_from_tomorrow = extract_result.get("start_from_tomorrow", False)

    # Xác định ngày bắt đầu và số ngày lấy
    start_offset = 1 if start_from_tomorrow else 0  # Bắt đầu từ ngày mai nếu "tới"
    start_date = CURRENT_DATE + timedelta(days=start_offset)
    days_to_fetch = min(days_requested, 5)  # Giới hạn API miễn phí
>>>>>>> Stashed changes

# user_input = "Các địa điểm du lịch nổi tiếng ở Đà Nẵng?"
# print(final_rag_chain.invoke({"question": user_input}))

<<<<<<< Updated upstream


from prompts.main_prompt_template import main_prompt_template

print(main_prompt_template)
=======
        if data.get("cod") != "200":
            logger.warning(f"API error: {data.get('message')}")
            return "<Ask> Không tìm thấy thông tin thời tiết cho thành phố này. Vui lòng kiểm tra lại tên."

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
            forecast_by_day[date]["rain"] = max(forecast_by_day[date]["rain"], item.get("rain", {}).get("3h", 0))

        # Lọc dữ liệu từ ngày bắt đầu
        weather_summary = []
        for i, (date, info) in enumerate(sorted(forecast_by_day.items())):
            forecast_date = datetime.strptime(date, "%Y-%m-%d")
            if forecast_date < start_date or i >= days_to_fetch:
                continue
            weather_summary.append({
                "date": date,
                "avg_temp": round(sum(info["temps"]) / len(info["temps"]), 1),
                "description": max(set(info["descriptions"]), key=info["descriptions"].count),
                "wind_speed": round(sum(info["wind_speeds"]) / len(info["wind_speeds"]), 1),
                "humidity": round(sum(info["humidities"]) / len(info["humidities"])),
                "rain": round(info["rain"], 1)
            })

        if not weather_summary:
            return "<Ask> Không có dữ liệu cho khoảng thời gian yêu cầu trong 5 ngày tới từ hôm nay."

    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return f"<Ask> Đã xảy ra lỗi khi tra cứu thời tiết: {str(e)}. Vui lòng thử lại."

    # Prompt tối ưu để sinh phản hồi
    weather_response_prompt = PromptTemplate(
        input_variables=["query", "weather_data", "days_requested", "start_from_tomorrow", "current_date"],
        template="""
        Bạn là một trợ lý du lịch chuyên nghiệp, thân thiện và giàu kinh nghiệm. Dựa trên dữ liệu thời tiết và câu hỏi của người dùng, hãy cung cấp phản hồi chi tiết, tự nhiên, hữu ích, mang tính cá nhân hóa cao cho mục đích du lịch.

        **Câu hỏi người dùng**: "{query}"
        **Dữ liệu thời tiết**: {weather_data}
        **Số ngày yêu cầu**: {days_requested}
        **Bắt đầu từ ngày mai**: {start_from_tomorrow}
        **Ngày hiện tại**: {current_date}

        **Hướng dẫn**:
        - Liệt kê dự báo thời tiết từng ngày (tối đa 5 ngày từ ngày bắt đầu):
          - Ngày tháng (định dạng DD/MM), nhiệt độ trung bình (°C), mô tả thời tiết, tốc độ gió (m/s), độ ẩm (%), lượng mưa (mm nếu có).
        - Nếu "start_from_tomorrow" là true, nhấn mạnh rằng đây là dự báo từ ngày mai trở đi.
        - Nếu số ngày yêu cầu > 5, giải thích nhẹ nhàng rằng chỉ có dữ liệu 5 ngày và gợi ý kiểm tra lại sau.
        - Gợi ý du lịch cụ thể dựa trên thời tiết:
          - Mưa > 0mm: Mang ô, đề xuất hoạt động trong nhà (bảo tàng, quán cà phê).
          - Gió > 7 m/s: Cảnh báo gió mạnh, tránh hoạt động ngoài trời như đi biển, leo núi.
          - Độ ẩm > 80%: Lưu ý cảm giác oi bức, khuyên mang nước hoặc mặc thoáng.
          - Nhiệt độ < 20°C: Gợi ý mặc ấm.
          - Nhiệt độ > 35°C: Đề xuất tránh nắng, dùng kem chống nắng.
        - Giữ giọng điệu tự nhiên, thân thiện, như một người bạn đồng hành đáng tin cậy.

        **Ví dụ**:
        - Input: "Thời tiết ở Hà Nội trong 2 ngày tới thế nào?"
          Weather Data: [
            {{"date": "2025-03-21", "avg_temp": 24, "description": "mây rải rác", "wind_speed": 3, "humidity": 70, "rain": 0}},
            {{"date": "2025-03-22", "avg_temp": 26, "description": "nắng", "wind_speed": 4, "humidity": 65, "rain": 0}}
          ]
          Output: "Dự báo thời tiết ở Hà Nội từ ngày mai trong 2 ngày tới:\n- 21/03: 24°C, mây rải rác, gió 3 m/s, độ ẩm 70% – Thời tiết dễ chịu, rất hợp để dạo quanh Hồ Gươm!\n- 22/03: 26°C, nắng, gió 4 m/s, độ ẩm 65% – Trời đẹp, tha hồ chụp ảnh ở phố cổ hay Văn Miếu!\nNhìn chung, thời tiết rất lý tưởng cho chuyến đi của bạn!"
        - Input: "Thời tiết ở Đà Nẵng trong 10 ngày tới thế nào?"
          Weather Data: [...5 ngày dữ liệu...]
          Output: "Dự báo thời tiết ở Đà Nẵng từ ngày mai trong 5 ngày tới (tôi chỉ có dữ liệu đến đó thôi, bạn có thể hỏi lại sau vài ngày nhé!):\n- 21/03: 28°C, mưa nhỏ, gió 6 m/s, độ ẩm 85%, mưa 2mm – Mang ô theo vì có mưa nhẹ, ghé bảo tàng sẽ hợp lý!\n[...]\nThời tiết ấm áp, thích hợp để khám phá nhưng cần chuẩn bị cho mưa nhỏ!"
        """
    )

    # Chain sinh phản hồi
    response_chain = weather_response_prompt | chatbot.llm_gemini | StrOutputParser()
    try:
        response = response_chain.invoke({
            "query": query,
            "weather_data": json.dumps(weather_summary, ensure_ascii=False),
            "days_requested": days_requested,
            "start_from_tomorrow": start_from_tomorrow,
            "current_date": CURRENT_DATE.strftime("%Y-%m-%d")
        })
        logger.info(f"Generated response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        # Phản hồi dự phòng
        response = f"Dự báo thời tiết ở {city}:\n"
        for day in weather_summary:
            rain_info = f", mưa {day['rain']}mm" if day['rain'] > 0 else ""
            date_str = datetime.strptime(day['date'], "%Y-%m-%d").strftime("%d/%m")
            response += f"- {date_str}: {day['avg_temp']}°C, {day['description']}, gió {day['wind_speed']} m/s, độ ẩm {day['humidity']}%{rain_info}.\n"
        if days_requested > 5:
            response += "Tôi chỉ có dữ liệu 5 ngày thôi, bạn quay lại hỏi thêm sau nhé!"
        return response

# Test
if __name__ == "__main__":
    from src.chatbot import Chatbot
    chatbot = Chatbot()
    queries = [
        "Thời tiết ở Hà Nội trong 2 ngày tới thế nào?",
        "Thời tiết ở Đà Nẵng hôm nay thế nào?",
        "Thời tiết ở Thành phố Hồ Chí Minh trong 10 ngày tới thế nào?"
    ]
    for query in queries:
        print(f"Query: {query}")
        print(f"Response: {weather_info_function(chatbot, query)}\n")
>>>>>>> Stashed changes
