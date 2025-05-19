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
from functools import partial


# Import các thư viện từ LangChain và các thư viện khác
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import Tool
from pydantic import Field
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field

# import prompt templates
from prompts.query_history_prompt_template import query_history_prompt_template
from prompts.location_info_prompt_template import location_info_prompt
from prompts.itinerary_planner_prompt_template import itinerary_planner_prompt_template
from prompts.weather_info_prompt_template import extract_info_prompt, weather_response_prompt
from prompts.price_search_prompt_template import price_prompt
from prompts.react_prompt import prompt_react

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
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def get_questions(chatbot, user_input):
    """Lấy danh sách câu hỏi từ query_generation_chain."""
    result = chatbot.query_generation_chain.invoke({"question": user_input})
    try:
        logger.info(f"\nKết quả từ query_generation_chain: {result}\n")
        return result["questions"]
    except (KeyError, TypeError) as e:
        logger.error(f"Lỗi khi xử lý output từ query_generation_chain: {e}")
        return [user_input]  # Trả về câu hỏi gốc nếu có lỗi

def format_input(retrieval_chain_rag_fusion, inputs):
    """Chuẩn bị dữ liệu đầu vào cho LLM."""
    context_result = retrieval_chain_rag_fusion.invoke(inputs)
    return {"retrieved_context": context_result, "question": inputs["question"]}

def generate_response(chatbot, user_input, prompt_template_for_query):
    """Sinh phản hồi dựa trên truy vấn và ngữ cảnh."""
    # Lấy danh sách câu hỏi từ query_generation_chain
    cleaned_query_generation = RunnableLambda(lambda x: get_questions(chatbot, x))

    retrieval_chain_rag_fusion = (
        cleaned_query_generation
        | chatbot.retriever.map()  # Áp dụng retriever cho từng câu hỏi trong danh sách
        | RunnableLambda(lambda results: reciprocal_rank_fusion(results))  # Gộp kết quả
    )

    document_retrieval_chain = retrieval_chain_rag_fusion.invoke({"question": user_input})
    # logger.info(f"\nGọi RAG Fusion chain với câu truy vấn: {user_input}\n {document_retrieval_chain}\n")

    formatted_prompt = RunnableLambda(lambda x: prompt_template_for_query.format(**x))

    final_rag_chain = (
        RunnableLambda(lambda x: format_input(retrieval_chain_rag_fusion, x))
        | formatted_prompt
        | chatbot.llm_gemini
        | StrOutputParser()
    )

    return final_rag_chain.invoke({"question": user_input})

""" -------------------------------------------------------------- """
"""     Tăng cường cho câu hỏi người dùng."""
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools.base import ArgsSchema

# Khởi tạo LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

""" ----------------------------------------------------- """
"""      Tăng cường ngữ cảnh cho câu hỏi."""
## Viết lại truy vấn
def rewrite_query(llm, query, history):
    chain = query_history_prompt_template | llm | JsonOutputParser()
    return chain.invoke({"query": query, "history": history})

""" ----------------------------------------------------- """
"""      Kiểm tra tính liên quan của câu hỏi."""
## Định nghĩa model cho việc trích xuất thông tin du lịch
class TravelInfoExtraction(BaseModel):
    is_clear: Optional[bool] = Field(
        None,
        description="True nếu câu hỏi đã rõ ràng không cần hỏi thêm thông tin, False nếu cần hỏi thêm thông tin."
    )
    missing_info: Optional[str] = Field(
        None,
        description="Thông tin còn thiếu trong câu hỏi để hỏi lại người dùng giúp câu hỏi rõ ràng hơn. Tránh nhập nhằng về ngữ nghĩa."
    )
    
## Công cụ: Trích xuất thông tin du lịch
def extract_travel_info_tool(llm, query):
    prompt = PromptTemplate(
        template="""
        Bạn là chuyên gia phân tích câu truy vấn du lịch. Nhiệm vụ của bạn là phân tích câu hỏi và xác định xem câu hỏi có rõ ràng hay không, có có cần hỏi thêm thông tin hay không.
        Truy vấn: {query}
        Nếu không đề cập tới thời gian, mặc định ngày hiện tại.
        Thời gian hiện tại: {current_time}
        """,
        input_variables=["query"]
    ).partial(current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    chain = prompt | llm.with_structured_output(TravelInfoExtraction)
    return chain.invoke({"query": query})

# Công cụ: Hỏi thông tin còn thiếu
def ask_missing_info_tool(llm, query, missing_info):
    prompt = PromptTemplate(
        template="""
        Người dùng đã hỏi: {query}
        Các thông tin còn thiếu là: {missing_info}
        Hãy đặt một câu hỏi lịch sự và rõ ràng để yêu cầu người dùng cung cấp các thông tin đó.
        """,
        input_variables=["query", "missing_info"]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "missing_info": missing_info})

# Hàm kiểm tra và xác thực truy vấn
def validate_query(llm, query: str, history: List) -> str:
    # Trích xuất thông tin du lịch
    travel_info = extract_travel_info_tool(llm, query)
    logger.info(f"\nKết quả từ extract_travel_info_tool: {travel_info}\n")
    
    # Nếu cần hỏi thêm thông tin, gọi hàm AskMissingInfoTool
    if not travel_info.is_clear:
        ask_missing_info = ask_missing_info_tool(llm, query, travel_info.missing_info)
        return {'ask_human': True, 'result': ask_missing_info}
    
    return {'ask_human': False, 'result': query}


""" ----------------------------------------------- """
"""    Xử lý câu hỏi không liên quan hoặc chào hỏi."""
from typing import Literal, Optional, Dict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

from pydantic import BaseModel, Field
from typing import Optional

class Relevant(BaseModel):
    is_relevant: bool = Field(
        description="True nếu câu hỏi liên quan trực tiếp đến du lịch (lên kế hoạch, điểm đến, phương tiện, ăn uống khi đi du lịch, lưu trú, visa, kinh nghiệm, tra cứu thời tiết, v.v.), "
                    "ngược lại False."
    )
    reason: Optional[str] = Field(
        description="Nếu is_relevant=False, giải thích ngắn gọn lý do tại sao không liên quan."
    )
    response: Optional[str] = Field(
        description="Nếu is_relevant=False, phản hồi thân thiện hướng người dùng quay lại chủ đề du lịch."
    )

def relevant_travel(llm_gemini, query):
    relevant_prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""
        Bạn là một trợ lý du lịch chuyên nghiệp, thân thiện. Nhiệm vụ của bạn là xác định xem câu hỏi của người dùng có thật sự liên quan đến du lịch hay không.

        - Định nghĩa "liên quan đến du lịch":
            + Lên kế hoạch chuyến đi: điểm đến, lịch trình, thời gian, ngân sách.  
            + Vận chuyển: máy bay, tàu, xe buýt, ô tô tự lái.  
            + Lưu trú: khách sạn, homestay, resort, thuê nhà dài/ngắn hạn.  
            + Ăn uống khi đi du lịch: món đặc sản, nhà hàng, quán địa phương nhưng trong bối cảnh “nên ăn khi du lịch tại…”.  
            + Thủ tục: visa, hải quan, bảo hiểm du lịch.  
            + Hoạt động, trải nghiệm: tham quan, tour, hướng dẫn viên, văn hoá, sự kiện.  
            + Thời tiết: tra cứu thông tin về thời tiết một ngày bất kì, ...
        
        - Không tính là liên quan:  
            + Chào hỏi xã giao ('Chào em!', 'Hôm nay thế nào?').  
            + Hỏi-đáp về lập trình, khoa học, thể thao…  

        Dưới đây là ví dụ:
        <examples>
        1. Câu hỏi: "Các món ăn nổi tiếng tại Đà Nẵng"  
        → is_relevant: true  
        (vì hỏi “ăn uống” trong bối cảnh địa điểm du lịch)  

        2. Câu hỏi: "Công thức làm phở bò"  
        → is_relevant: false  
        reason: "Hỏi chung về nấu ăn, không đặt trong bối cảnh du lịch"  
        response: "Mình chuyên hỗ trợ về du lịch, bạn có muốn tìm hiểu địa điểm ẩm thực khi đi du lịch không?"  

        3. Câu hỏi: "Chào bạn, bạn khỏe không?"  
        → is_relevant: false  
        reason: "Chào hỏi xã giao, không liên quan du lịch, phản hồi lại nhẹ nhàng"  
        response: "Chào bạn! Mình là trợ lý du lịch, bạn có thắc mắc gì về chuyến đi không?"  
        </examples>

        User question: "{query}"

        Hãy chỉ trả về JSON hợp lệ theo schema Pydantic `Relevant`, không thêm text nào khác.
        """ 
        )
    chain = relevant_prompt_template | llm_gemini.with_structured_output(Relevant)
    return chain.invoke({"query": query})


# res = not_relevant_function("Cho tôi thông tin về ronaldo")
# print(res)  # Chào bạn, hôm nay trời đẹp nhỉ?

""" --------------------------------- """
"""Xử lý các câu hỏi liên quan đến địa điểm, thông tin địa điểm."""
def location_info_function(chatbot, query):
    return generate_response(chatbot, query, location_info_prompt)


""" --------------------------------- """
"""Xử lý các câu hỏi liên quan đến lập kế hoạch du lịch."""
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
    except Exception as e:
        return "Vui lòng cung cấp thông tin rõ ràng hơn (địa điểm, thời gian, sở thích)."

    if not extracted_info.get("destination") or not extracted_info.get("duration"):
        return "Bạn muốn đi đâu và trong bao lâu?"

    destination = extracted_info["destination"]
    duration = extracted_info["duration"]
    print("========================Thông tin bổ sung=============================")
    # Lấy thông tin bổ sung
    try:
        date_time = chatbot.print_date_time()
        weather_query = f"Thời tiết ở {destination} trong {duration} bắt đầu từ ngày {date_time}"
        weather_data = chatbot.executor.invoke({"input": weather_query})
        weather_data = weather_data.get("output")
        if "<Ask>" in weather_data:
            weather_data = "Thời tiết bình thường."

        location_query = f"Thông tin về {destination}"
        location_data = chatbot.executor.invoke({"input": location_query})
        location_data = location_data.get("output")
        if "<Ask>" in location_data:
            location_data = f"{destination} là một điểm đến phổ biến."

        price_query = f"Giá dịch vụ du lịch ở {destination} (khách sạn, vé tham quan, ăn uống)"
        price_data = chatbot.executor.invoke({"input": price_query})
        price_data = price_data.get("output")
        if "<Ask>" in price_data:
            price_data = "Giá cả trung bình."

    except Exception as e:
        weather_data = "Thời tiết bình thường."
        location_data = f"{destination} là một điểm đến phổ biến."
        price_data = "Giá cả trung bình."

    # Sinh lịch trình
    response_chain = itinerary_planner_prompt_template | chatbot.llm_gemini | StrOutputParser()
    retrieved_context = chatbot.retriever.invoke(query)
    all_contents = [doc.page_content for doc in retrieved_context]
    try:
        response = response_chain.invoke({
            "query": query,
            "retrieved_context": all_contents,
            "weather_data": weather_data,
            "location_data": location_data,
            "price_data": price_data
        })
        return response
    except Exception as e:
        return f"Lỗi khi lập kế hoạch: {str(e)}. Vui lòng thử lại."

""" ------------------------------------------------------------ """
"""             Hàm tìm kiếm thông tin từ Tavily.           """
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# Tạo một công cụ tìm kiếm với Tavily
tavily_tool = TavilySearchResults(
    max_results=3,
    include_answer=True,       # sẽ trả về câu trả lời/tóm tắt tự động
    include_raw_content=False,  # kèm luôn nội dung thô
    search_depth="advanced",
)

class EnhancerStruct(BaseModel):
    query: str = Field(
        description="Câu truy vấn gốc từ người dùng."
    )
    version_1: str = Field(
        description="Phiên bản 1 của câu truy vấn đã được tối ưu hóa."
    )
    version_2: str = Field(
        description="Phiên bản 2 của câu truy vấn đã được tối ưu hóa."
    )
    version_3: str = Field(
        description="Phiên bản 3 của câu truy vấn đã được tối ưu hóa."
    )
    

# Nhiệm vụ tạo nhiều phiên bản câu truy vấn chất lượng
def enhancer_func_tool(llm, query: str) -> str:
    """
    Enhancer agent node that improves and clarifies user queries.
    Takes the original user input and transforms it into a more precise,
    actionable request before passing it to the supervisor.
    """
    template = (
        "Bạn là một Chuyên gia Tinh chỉnh Truy vấn, có nhiệm vụ biến các câu truy vấn của người dùng thành nhiều phiên bản tối ưu hóa cho công cụ tìm kiếm như Tavily Search. Nhiệm vụ của bạn bao gồm:\n\n"
        "1. Phân tích câu truy vấn gốc để xác định ý định chính và yêu cầu cốt lõi.\n"
        "2. Giải quyết các điểm mơ hồ bằng cách đưa ra các giả định hợp lý hoặc diễn đạt lại cho rõ ràng.\n"
        "3. Mở rộng các khía cạnh chưa được phát triển đầy đủ của truy vấn bằng ngôn ngữ mô tả hoặc chi tiết liên quan.\n"
        "4. Tái cấu trúc truy vấn để đảm bảo rõ ràng và khả năng tìm kiếm.\n"
        "5. Đảm bảo mọi thuật ngữ kỹ thuật được đặt trong ngữ cảnh phù hợp.\n\n"
        "Quan trọng: Không được yêu cầu thêm thông tin từ người dùng. Hãy sử dụng chuyên môn của bạn để tạo ra các phiên bản truy vấn toàn diện và hiệu quả nhất có thể. Nếu truy vấn thiếu chi tiết cụ thể, hãy nâng cấp nó bằng ngôn ngữ mô tả hoặc dự đoán nhu cầu phổ biến của người dùng.\n\n"
        
        "Kết quả: Tạo ra ba phiên bản khác nhau của truy vấn:\n"
        "- **Phiên bản 1 (Chi tiết):** Rất chi tiết và cụ thể, bao gồm ngôn ngữ mô tả, từ khóa liên quan và cụm từ để thu hẹp kết quả tìm kiếm. Nếu phù hợp, dự đoán nhu cầu cụ thể của người dùng (ví dụ: với 'thời tiết', bao gồm nhiệt độ, mưa).\n"
        "- **Phiên bản 2 (Trung bình):** Chi tiết ở mức vừa phải, cân bằng giữa độ cụ thể và độ bao quát. Diễn đạt lại truy vấn sao cho rõ và ngắn gọn nhưng vẫn giữ được tính liên quan.\n"
        "- **Phiên bản 3 (Tổng quát):** Tổng quát và rộng, tập trung vào ý định cốt lõi của truy vấn với ít chi tiết bổ sung.\n\n"
        "Mỗi phiên bản phải khác biệt và được thiết kế để tối ưu hóa tính liên quan của kết quả tìm kiếm khi sử dụng với Tavily Search.\n\n"
        
        "User query: {query}\n\n"
        
        "Thời gian hiện tại: {current_time}\n\n"
    )
    
    prompt = PromptTemplate(
        input_variables=["query"],
        template=template
    ).partial(current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    chain = prompt | llm.with_structured_output(EnhancerStruct)
    
    return chain.invoke({"query": query})

# Hàm tìm kiếm với Tavily
def search_agent(Chatbot, query: str):
    llm = Chatbot.llm_gemini
    enhancer_tool = Tool(
        name="enhancer_tool",
        func=lambda input: enhancer_func_tool(llm, input),
        description=(
            "MUST be called FIRST: trả về object với 3 phiên bản truy vấn."
        )
    )
    
    search_tool = Tool(
        name="tavily_tool",
        func=lambda input: tavily_tool.invoke({"query": input}),
        description=(
            "Chạy tìm kiếm thông tin với từng kết quả đầu vào."
        )
    )
    
    tools = [search_tool, enhancer_tool]
    try:
        agent = create_react_agent(
            llm=llm,
            tools=tools,    
            prompt = prompt_react
        )
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True
        ) 
        return agent_executor.invoke({"input": query})
    except Exception as e:
        logger.error(f"Error with Tavily search: {str(e)}")
        return f"Lỗi khi tìm kiếm: {str(e)}"
    
# print(search_agent(llm, "giá vé máy bay từ Đà Nẵng đi Hà Nội ngày mai?"))

""" ---------------------------------------------------------------------------"""
"""     Hàm tra cứu và sinh phản hồi thời tiết chi tiết cho chatbot du lịch."""
class ExtractInFoTool(BaseModel):
    city: Optional[str] = Field(
        None,
        description="Tên thành phố cần tra cứu thời tiết. Lưu ý viết thường không dấu. ví dụ: 'ho chi minh'"
    )
    date: List[str] = Field(
        description="Các ngày cần tra cứu thời tiết. Nếu không có, mặc định là ngày hiện tại."
    )
    
    amount: int = Field(
        default=1,
        description="Số lượng ngày cần tra cứu thời tiết. Mặc định là 1."
    )

# Hàm tra cứu và sinh phản hồi thời tiết
def weather_info_function(llm, query):
    # llm = Chatbot.llm_gemini
    BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

    # Chain trích xuất
    extract_chain = extract_info_prompt | llm.with_structured_output(ExtractInFoTool)
    try:
        extract_result = extract_chain.invoke({"query": query})
        logger.info(f"Extracted result: {extract_result}")
    except Exception as e:
        logger.error(f"Error extracting info: {str(e)}")
        return "Tôi không hiểu yêu cầu của bạn. Vui lòng thử lại."

    city = extract_result.city

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
            return "Không tìm thấy thông tin thời tiết cho thành phố này."

    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return f"Lỗi khi tra cứu thời tiết: {str(e)}. Vui lòng thử lại."

    # Chain sinh phản hồi (truyền JSON thô)
    response_chain = weather_response_prompt | llm | StrOutputParser()
    try:
        response = response_chain.invoke({
            "query": query,
            "weather_data": json.dumps(data, ensure_ascii=False),  # Truyền toàn bộ JSON thô
            "requested_dates": extract_result.date,
            "amount": extract_result.amount,    
        })
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Lỗi khi xử lý dữ liệu thời tiết: {str(e)}. Vui lòng thử lại."


def agent_weather(Chatbot, query):
    llm = Chatbot.llm_gemini
    weather_tool = Tool(
        name="weather_tool",
        func=lambda input: weather_info_function(llm, input),
        description="Cung cấp thông tin thời tiết cho một thành phố cụ thể trong một khoảng thời gian nhất định."
    )
    
    tools = [weather_tool]
    """Hàm xử lý thời tiết với agent."""
    try:
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt= prompt_react  # Sử dụng prompt từ langchain hub
        )
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True
        )
        return agent_executor.invoke({"input": query})
    except Exception as e:
        logger.error(f"Error with weather agent: {str(e)}")
        return f"Lỗi khi xử lý thời tiết: {str(e)}"

# print(agent_weather(llm, "Thời tiết ở Đà Nẵng 10 ngày tới là gì?"))