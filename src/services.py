from .utils import *
import os
import sys
import logging
import requests
from datetime import datetime
import json
import time

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from pydantic import Field


# import prompt templates
from prompts.query_history_prompt_template import query_history_prompt_template
from prompts.location_info_prompt_template import location_info_prompt
from prompts.itinerary_planner_prompt_template import itinerary_planner_prompt
from prompts.weather_info_prompt_template import extract_info_prompt, weather_response_prompt
from prompts.price_search_prompt_template import travel_info_prompt_template


from dotenv import load_dotenv
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Services")

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
def itinerary_planner_function(chatbot, query):
    return generate_response(chatbot, query, itinerary_planner_prompt)


""" --------------------------------- """
"""Hàm tra cứu và sinh phản hồi thời tiết chi tiết cho chatbot du lịch."""
def weather_info_function(chatbot, query):
    # API_KEY = 'b13f85eb589c453522bb1322a6763a8d'
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
"""Tìm kiếm thông tin giá vé, dịch vụ du lịch trên internet."""
class CustomGoogleSearchAPIWrapperContent(GoogleSearchAPIWrapper):
    def __init__(self, google_api_key=None, google_cse_id=None, **kwargs):
        """
            Bỏ key trong file .env
        """
        google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        google_cse_id = google_cse_id or os.getenv("GOOGLE_CSE_ID")

        kwargs.setdefault("google_api_key", google_api_key)
        kwargs.setdefault("google_cse_id", google_cse_id)

        super().__init__(**kwargs)

    def get_page_content(self, url):
        """Trích xuất nội dung chính từ trang web. Nếu fail, trả về None."""
        try:
            loader = WebBaseLoader(
                url,
                requests_per_second = 1
            )
            docs = loader.load()
            if not docs:
                return None
            content = docs[0].page_content
            # Xử lý lỗi font/encoding nếu cần
            try:
                content.encode("utf-8").decode("utf-8")
            except UnicodeDecodeError:
                content = content.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            return content[:3000]  # Giới hạn độ dài để tránh input quá tải
        except Exception as e:
            print(f"[ERROR] Không thể tải nội dung từ {url}: {e}")
            return None

    def run(self, query, num_results=3):
        search_results = self.results(query, num_results=num_results)
        if not search_results:
            return {
                "answer": "Không tìm thấy kết quả phù hợp.",
                "references": [],
            }

        content_list = []
        references = []

        for result in search_results:
            url = result.get("link")
            if not url:
                continue
            references.append(url)

            page_text = self.get_page_content(url)
            if page_text:
                content_list.append(f"{result['title']} ({url}):\n{page_text}\n")
            else:
                content_list.append(f"{result['title']} ({url}): Không thể lấy dữ liệu.\n")

        full_content = "\n\n".join(content_list)

        return {
            "answer": full_content,
            "references": references,
        }

# Hàm gọi chatbot với thông tin tìm kiếm
search_tool = CustomGoogleSearchAPIWrapperContent()

def price_search_function(chatbot, query):
    query += " mới nhất"
    document = search_tool.run(query)

    document_str = json.dumps(document["answer"], ensure_ascii=False, indent=2)
    references_str = "\n".join(document["references"])

    chain = travel_info_prompt_template | chatbot.llm_gemini | StrOutputParser()

    response = chain.invoke({
        "input": query,
        "documents": document_str,
        "references": references_str
    })
    return response
