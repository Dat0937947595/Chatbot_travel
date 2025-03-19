from .utils import *
import logging
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import PromptTemplate

# import prompt templates
from prompts.query_history_prompt_template import query_history_prompt_template
from prompts.location_info_prompt_template import location_info_prompt
from prompts.itinerary_planner_prompt_template import itinerary_planner_prompt

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
    # try:
    #     api_key = os.getenv("OPENWEATHER_API_KEY")
    #     location = extract_location(query)
    #     url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    #     response = requests.get(url).json()
    #     return f"Thời tiết tại {location}: {response['main']['temp']}°C, {response['weather'][0]['description']}."
    # except Exception as e:
    #     return f"Không thể lấy thông tin thời tiết: {str(e)}"
    
    return "Thời tiết tại Đà Nẵng: 30°C, nắng nhẹ."

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














