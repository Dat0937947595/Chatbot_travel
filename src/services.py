from .utils import *
import logging
import requests
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import PromptTemplate
from datetime import datetime
import json

# import prompt templates
from prompts.query_history_prompt_template import query_history_prompt_template
from prompts.location_info_prompt_template import location_info_prompt
from prompts.itinerary_planner_prompt_template import itinerary_planner_prompt
from prompts.weather_info_prompt_template import extract_info_prompt, weather_response_prompt


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Services")

def query_history(chatbot, query):
    chat_history = chatbot.memory.load_memory_variables({})["chat_history"]
    query_history_chain = (
        query_history_prompt_template
        | chatbot.llm_gemini
        | StrOutputParser()
    )
    return query_history_chain.invoke({"history": chat_history, "query": query})

def generate_response(chatbot, user_input, prompt_template_for_query):
    """Sinh phản hồi dựa trên truy vấn và ngữ cảnh, sử dụng danh sách câu hỏi từ query_generation_chain."""
    logger.info(f"\nSinh ra câu trả lời từ câu truy vấn: {user_input}\n")
    
    # Lấy danh sách câu hỏi từ query_generation_chain
    def get_questions(x):
        result = chatbot.query_generation_chain.invoke({"question": x})
        try:
            
            logger.info(f"\nKết quả từ query_generation_chain: {result}\n")
            # Lấy danh sách câu hỏi từ key "questions"
            return result["questions"]
        except (KeyError, TypeError) as e:
            logger.error(f"Lỗi khi xử lý output từ query_generation_chain: {e}")
            return [x]  # Trả về câu hỏi gốc nếu có lỗi
    
    cleaned_query_generation = RunnableLambda(get_questions)
    
    retrieval_chain_rag_fusion = (
        cleaned_query_generation
        | chatbot.retriever.map()  # Áp dụng retriever cho từng câu hỏi trong danh sách
        | RunnableLambda(lambda results: reciprocal_rank_fusion(results))  # Gộp kết quả
    )
    
    def format_input(inputs):
        context_result = retrieval_chain_rag_fusion.invoke(inputs)
        return {"retrieved_context": context_result, "question": inputs["question"]}
    
    formatted_prompt = RunnableLambda(lambda x: prompt_template_for_query.format(**x))
    
    final_rag_chain = (
        RunnableLambda(format_input)
        | formatted_prompt
        | chatbot.llm_gemini
        | StrOutputParser()
    )
    
    return final_rag_chain.invoke({"question": user_input})

def location_info_function(chatbot, query):
    return generate_response(chatbot, query, location_info_prompt)

def itinerary_planner_function(chatbot, query):
    return generate_response(chatbot, query, itinerary_planner_prompt)

def greetings_function(chatbot, user_input):
    """Sinh phản hồi tự nhiên cho các câu chào hỏi hoặc giao tiếp xã giao bằng LLM Gemini."""
    # Tạo prompt dưới dạng PromptTemplate
    greeting_prompt_template = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
        Bạn là một trợ lý thân thiện. Hãy trả lời câu chào hỏi hoặc giao tiếp xã giao từ người dùng một cách tự nhiên, lịch sự dựa trên câu hỏi và ngữ cảnh lịch sử hội thoại.
        Lịch sử hội thoại: {chat_history}
        Câu hỏi: {question}
        Trả về phản hồi ngắn gọn, thân thiện.
        """
    )
    
    # Tạo chain: PromptTemplate -> LLM Gemini -> string output
    greeting_chain = (
        greeting_prompt_template
        | chatbot.llm_gemini
        | StrOutputParser()
    )
    
    # Lấy lịch sử hội thoại từ memory
    chat_history = chatbot.memory.load_memory_variables({})["chat_history"]
    
    # Gọi chain với input là dict chứa chat_history và question",
    response = greeting_chain.invoke({"chat_history": chat_history, "question": user_input})
    return response

def not_relevant_function(chatbot, user_input):
    """Phản hồi khi câu hỏi không liên quan đến du lịch."""
    # Tạo prompt dưới dạng PromptTemplate
    not_relevant_prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""
        Bạn là một trợ lý du lịch thông minh, chỉ hỗ trợ các câu hỏi về du lịch hoặc giao tiếp xã giao.
        Câu hỏi của người dùng: {question}
        Hãy từ chối nhẹ nhàng nếu câu hỏi không liên quan đến du lịch, và gợi ý họ hỏi về du lịch.
        Trả về phản hồi ngắn gọn.
        """
    )
    
    # Tạo chain: PromptTemplate -> LLM Gemini -> string output
    not_relevant_chain = (
        not_relevant_prompt_template
        | chatbot.llm_gemini
        | StrOutputParser()
    )
    
    # Gọi chain với input là dict chứa question
    response = not_relevant_chain.invoke({"question": user_input})
    return response

def weather_info_function(chatbot, query):
    """Hàm tra cứu và sinh phản hồi thời tiết chi tiết cho chatbot du lịch."""
    API_KEY = 'b13f85eb589c453522bb1322a6763a8d'
    BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

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

def price_search_function(chatbot, query):
    # try:
    #     api_key = os.getenv("AMADEUS_API_KEY")
    #     origin, destination, date = extract_flight_info(query)
    #     url = f"https://api.amadeus.com/v2/shopping/flight-offers?origin={origin}&destination={destination}&departureDate={date}&apiKey={api_key}"
    #     response = requests.get(url).json()
    #     price = response['data'][0]['price']['total']
    #     return f"Giá vé từ {origin} đến {destination} ngày {date}: {price} VND"
    # except Exception as e:
    #     return f"Không thể tìm giá vé: {str(e)}"
    return "Giá vé từ Hà Nội đến Đà Nẵng ngày 20/3/2025: 1.2 triệu VND."














