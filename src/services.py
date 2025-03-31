import os
import sys
# Thêm thư mục gốc (CHATBOT_TRAVEL) vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các thư viện cần thiết
import logging
import requests
from datetime import datetime
import json
import time
from tavily import TavilyClient

# Import các thư viện từ LangChain và các thư viện khác
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from pydantic import Field


# import prompt templates
from prompts.query_history_prompt_template import query_history_prompt_template
from prompts.location_info_prompt_template import location_info_prompt
from prompts.itinerary_planner_prompt_template import itinerary_planner_prompt_template
from prompts.weather_info_prompt_template import extract_info_prompt, weather_response_prompt
from prompts.price_search_prompt_template import price_prompt

# import utils
from src.utils import *

# Load biến môi trường
from dotenv import load_dotenv
load_dotenv()

# print(os.getenv("GOOGLE_API_KEY"))

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Services")

# Khởi tạo Tavily Client
tavily_client = TavilyClient(api_key="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")

"""Sinh phản hồi dựa trên truy vấn và ngữ cảnh, sử dụng danh sách câu hỏi từ query_generation_chain."""
def generate_response(chatbot, user_input, prompt_template_for_query):
    # logger.info(f"\nSinh ra câu trả lời từ câu truy vấn: {user_input}\n")
    
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
    
    document_retrieval_chain = retrieval_chain_rag_fusion.invoke({"question": user_input})
    logger.info(f"\nGọi RAG Fusion chain với câu truy vấn: {user_input}\n {document_retrieval_chain}\n")
    
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


"""Tinh chỉnh truy vấn, trả lời trực tiếp, hoặc hỏi lại nếu thiếu ngữ cảnh."""
def context_enhancer_function(chatbot, query):
    chat_history = chatbot.memory.load_memory_variables({})["chat_history"]
    query_history_chain = (
        query_history_prompt_template
        | chatbot.llm_gemini
        | JsonOutputParser()
    )
    result = query_history_chain.invoke({"history": chat_history, "query": query})
    
    # Trường hợp trả lời trực tiếp
    if "response" in result:
        return result["response"]
    # Trường hợp tinh chỉnh truy vấn
    elif "refined_query" in result:
        return result["refined_query"]
    # Trường hợp hỏi lại
    else:
        missing_info = result.get("missing_info", "Vui lòng cung cấp thêm thông tin (địa điểm, thời gian, v.v.).")
        return f"<Ask> {missing_info}"
    
    
""" --------------------------------- """
"""Sinh phản hồi tự nhiên cho các câu chào hỏi hoặc giao tiếp xã giao bằng LLM Gemini."""
def greetings_function(chatbot, user_input):
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


""" --------------------------------- """
"""Xử lý câu hỏi không liên quan hoặc chào hỏi."""
def not_relevant_function(chatbot, query):
    not_relevant_prompt_template = PromptTemplate(
        input_variables=["chat_history", "query"],
        template="""
        Bạn là một trợ lý du lịch thân thiện. Nếu câu hỏi không liên quan đến du lịch, từ chối nhẹ nhàng và gợi ý hỏi về du lịch. Nếu là chào hỏi, trả lời tự nhiên.
        Lịch sử trò chuyện: {chat_history}
        Câu hỏi: {query}
        Trả về phản hồi ngắn gọn, thân thiện.
        """
    )
    chain = not_relevant_prompt_template | chatbot.llm_gemini | StrOutputParser()
    chat_history = chatbot.memory.load_memory_variables({})["chat_history"]
    return chain.invoke({"chat_history": chat_history, "query": query})


""" --------------------------------- """
"""Xử lý các câu hỏi liên quan đến địa điểm, thông tin địa điểm."""
def location_info_function(chatbot, query):
    return generate_response(chatbot, query, location_info_prompt)


""" --------------------------------- """
"""Xử lý các câu hỏi liên quan đến lập kế hoạch du lịch."""
# Hàm lập kế hoạch tối ưu
def itinerary_planner_function(chatbot, query):
    """Lập kế hoạch chuyến đi chi tiết, tích hợp thông tin từ các tool khác."""
    
    extract_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        Trích xuất thông tin từ câu hỏi để lập kế hoạch du lịch:
        - Thời gian chuyến đi (số ngày, số đêm nếu có).
        - Điểm đến.
        - Sở thích (nghỉ dưỡng, văn hóa, ẩm thực, phiêu lưu, gia đình, ...).
        - Ngân sách (cao cấp, trung bình, tiết kiệm - nếu có).
        - Phương tiện di chuyển (nếu có).
        Câu hỏi: "{query}"
        Trả về JSON: 
        {{"duration": "<số ngày/ngày+đêm>", "destination": "<điểm đến>", "preferences": "<sở thích>", "budget": "<ngân sách>", "transport": "<phương tiện>"}}
        Nếu thiếu thông tin, để giá trị là null.
        """
    )
    
    extract_chain = extract_prompt | chatbot.llm_gemini | JsonOutputParser()
    try:
        extracted_info = extract_chain.invoke({"query": query})
        logger.info(f"Extracted info: {extracted_info}")
    except Exception as e:
        logger.error(f"Error extracting info: {str(e)}")
        return "<Ask> Vui lòng cung cấp thông tin rõ ràng hơn (địa điểm, thời gian, sở thích)."

    if not extracted_info.get("destination") or not extracted_info.get("duration"):
        return "<Ask> Bạn muốn đi đâu và trong bao lâu?"

    destination = extracted_info["destination"]
    duration = extracted_info["duration"]
    preferences = extracted_info.get("preferences", "khám phá chung")
    budget = extracted_info.get("budget", "trung bình")
    transport = extracted_info.get("transport", "taxi/xe máy")

    # Lấy thông tin bổ sung
    try:
        weather_query = f"Thời tiết ở {destination} trong {duration}"
        weather_data = chatbot.executor.invoke({"input": weather_query})
        if "<Ask>" in weather_data:
            weather_data = "Thời tiết bình thường."

        location_query = f"Thông tin về {destination}"
        location_data = chatbot.executor.invoke({"input": location_query})
        if "<Ask>" in location_data:
            location_data = f"{destination} là một điểm đến phổ biến."

        price_query = f"Giá dịch vụ du lịch ở {destination} (khách sạn, vé tham quan, ăn uống)"
        price_data = chatbot.executor.invoke({"input": price_query})
        if "<Ask>" in price_data:
            price_data = "Giá cả trung bình."

    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        weather_data = "Thời tiết bình thường."
        location_data = f"{destination} là một điểm đến phổ biến."
        price_data = "Giá cả trung bình."

    # Sinh lịch trình
    response_chain = itinerary_planner_prompt_template | chatbot.llm_gemini | StrOutputParser()
    try:
        response = response_chain.invoke({
            "query": query,
            "retrieved_context": chatbot.retriever.invoke(query),
            "weather_data": weather_data,
            "location_data": location_data,
            "price_data": price_data
        })
        return response
    except Exception as e:
        logger.error(f"Error generating itinerary: {str(e)}")
        return f"Lỗi khi lập kế hoạch: {str(e)}. Vui lòng thử lại."

"""" --------------------------------- """
"""Lấy thời gian hiện tại và múi giờ của các địa điểm du lịch."""
def get_time_function():
    # Dùng thư viện time để lấy thời gian hiện tại
    current_time = time.strftime("%H:%M:%S", time.localtime())
    current_date = time.strftime("%Y-%m-%d", time.localtime())
    current_timezone = time.tzname[0]  # Lấy tên múi giờ hiện tại
    
    # Tạo phản hồi
    response = f"Thời gian hiện tại là {current_time} ngày {current_date}, múi giờ: {current_timezone}."
    
    return response

""" --------------------------------- """
"""Hàm tra cứu và sinh phản hồi thời tiết chi tiết cho chatbot du lịch."""
def weather_info_function(chatbot, query):
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

    # Xác định ngày hiện tại (hôm nay)
    # Trong thực tế, sử dụng: current_date = datetime.now()
    current_date = datetime.now()
    current_date_str = current_date.strftime("%Y-%m-%d")

    # Gọi API OpenWeatherMap
    try:
        params = {
            "q": f"{city},VN",
            "appid": os.getenv("OPENWEATHERMAP_API_KEY"),
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
                    "rain": 0  # Khởi tạo lượng mưa bằng 0
                }
            forecast_by_day[date]["temps"].append(item["main"]["temp"])
            forecast_by_day[date]["descriptions"].append(item["weather"][0]["description"])
            forecast_by_day[date]["wind_speeds"].append(item["wind"]["speed"])
            forecast_by_day[date]["humidities"].append(item["main"]["humidity"])
            # Cộng dồn lượng mưa trong ngày
            forecast_by_day[date]["rain"] += item.get("rain", {}).get("3h", 0)  # Sửa từ max thành +=

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
                "rain": round(info["rain"], 1)  # Lượng mưa đã được cộng dồn
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
            "days_requested": days_requested,
            "current_date": current_date_str  # Truyền ngày hiện tại vào prompt
        })
        logger.info(f"Generated response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        # Phản hồi dự phòng
        response = f"Dự báo thời tiết ở {city}:\n"
        for day in weather_summary:
            # Ánh xạ ngày trong phản hồi dự phòng
            day_date = datetime.strptime(day['date'], "%Y-%m-%d")
            delta = (day_date - current_date).days
            if delta == 0:
                day_label = "Hôm nay"
            elif delta == 1:
                day_label = "Ngày mai"
            elif delta == 2:
                day_label = "Ngày kia"
            else:
                day_label = f"ngày {day_date.strftime('%d/%m')}"
            rain_info = f", mưa {day['rain']}mm" if day['rain'] > 0 else ""
            response += f"- {day_label}: {day['avg_temp']}°C, {day['description']}, gió {day['wind_speed']} m/s, độ ẩm {day['humidity']}%{rain_info}.\n"
        if days_requested > 5:
            response += "Tôi chỉ có dữ liệu 5 ngày thôi, bạn quay lại hỏi thêm sau nhé!"
        return response


""" --------------------------------- """
# Hàm tìm kiếm với Tavily
def tavily_search(query: str) -> str:
    try:
        results = tavily_client.search(query=query, search_depth="advanced", max_results=5)
        if not results or "results" not in results:
            return "Không tìm thấy kết quả phù hợp."

        response = []
        for item in results["results"]:
            title = item.get("title", "Không có tiêu đề")
            snippet = item.get("content", "Không có mô tả")
            url = item.get("url", "Không có đường dẫn")
            published_date = item.get("published_date", "Không rõ ngày xuất bản")
            response.append(f"**{title}** (Ngày: {published_date})\n{snippet}\n[Xem chi tiết]({url})")
        
        return "\n\n".join(response)
    except Exception as e:
        logger.error(f"Error with Tavily search: {str(e)}")
        return f"Lỗi khi tìm kiếm: {str(e)}"

# Hàm tìm kiếm giá vé, dịch vụ du lịch
def price_search_function(chatbot, query: str) -> str:
    # Thêm "mới nhất" để đảm bảo thông tin cập nhật
    search_query = f"{query} mới nhất"
    search_results = tavily_search(search_query)
    
    response_chain = price_prompt | chatbot.llm_gemini | StrOutputParser()
    try:
        response = response_chain.invoke({
            "query": query,
            "search_results": search_results
        })
        return response
    except Exception as e:
        logger.error(f"Error generating price response: {str(e)}")
        return f"Không thể tìm thông tin giá chính xác cho '{query}'. Bạn có thể thử lại hoặc kiểm tra trực tiếp qua các trang web du lịch."