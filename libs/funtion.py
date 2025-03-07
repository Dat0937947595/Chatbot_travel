'''
    Dùng đề xử lý các vấn đề liên quan tới câu hỏi người dùng:
        - Viết lại câu hỏi (nếu cần thiết)
        - Hỏi lại câu hỏi (nếu cần thiết)
        - Loại các câu hỏi không liên quan
        - ...
'''
### Import thư viện
import logging
import os
import dotenv
from libs.prompt_template import *
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory

''' -------------------------------------- '''

### Load các biến môi trường
dotenv.load_dotenv("api.env")
api_key_llm = os.getenv("GROQ_API_KEY")
''' -------------------------------------- '''

### Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Question Processing")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True  # Trả về dạng danh sách tin nhắn
)
''' -------------------------------------- '''

history_all = ""
llm_all = ChatGroq(
            model_name="deepseek-r1-distill-qwen-32b",
            temperature=0.7,
            api_key=api_key_llm
)

### Hàm refine query dựa trên lịch sử hội thoại
def refine_query(user_query):
    """
    Refine câu hỏi của người dùng dựa trên lịch sử hội thoại, sử dụng prompt mới với phương pháp Chain of Thought (CoT).
    """
    try:
        # Thực hiện refine câu hỏi
        refined_query_chain = (
            query_refinement_prompt_template
            | llm_all  # Sử dụng Large Language Model để sinh output
            | StrOutputParser()  # Parser để chuyển đổi output về dạng string
        )
        
        print(history_all)
        refined_query = refined_query_chain.invoke({"chat_history": history_all, "user_query": user_query})
    except Exception as e:
        logger.error(f"Lỗi khi refine query: {e}")
        # Trường hợp lỗi, giữ nguyên câu hỏi gốc
        refined_query = user_query
    
    return refined_query



def summary_history(conversation):
    summary_history_chain = (
        summary_history_prompt_template
        | llm_all  # Sử dụng Large Language Model để sinh output
        | StrOutputParser()  # Parser để chuyển đổi output về dạng string
    )
    response = summary_history_chain.invoke({"history": history_all, "new_conversation": conversation})
    return response

def query_history(query):
    query_history_chain =(
        query_history_prompt_template
        | llm_all
        | StrOutputParser() 
    )
    response = query_history_chain.invoke({"history": history_all, "query": query})
    return response

# Định nghĩa tool TravelAgent
tool_memory_agent = Tool(
    name="MemoryAgent",
    func=query_history,
    description="Tìm kiếm thông tin mà người dùng cung cấp hoặc trong cuộc trao đổi trong lịch sử trò chuyện."
)

# Định nghĩa tool HotelSearchAgent
tool_travel_agent = Tool(
    name="TravelAgent",
    func=refine_query,
    description="Tìm kiếm khách sạn dựa trên địa điểm, ngày nhận phòng và ngày trả phòng."
)

# # Chạy thử
# test_flight = tool_travel_agent.run({"origin": "SGN", "destination": "HAN", "date": "2025-03-10"})
# test_hotel = tool_hotel_agent.run({"location": "Hà Nội", "check_in": "2025-03-10", "check_out": "2025-03-15"})


# Khởi tạo React Agent
react_agent = create_react_agent(
    llm=llm_all,
    tools=[tool_memory_agent, tool_travel_agent],
    prompt=main_prompt
)

# Tạo AgentExecutor
agent_search_executor = AgentExecutor(
    agent=react_agent,
    tools=[tool_memory_agent, tool_travel_agent],
    memory=memory,
    verbose=True,
    handle_parsing_errors=False
)






