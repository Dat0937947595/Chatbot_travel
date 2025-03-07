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
''' -------------------------------------- '''

### Load các biến môi trường
dotenv.load_dotenv("api.env")
''' -------------------------------------- '''
api_key_llm = os.getenv("GROQ_API_KEY")
### Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Question Processing")

''' -------------------------------------- '''

### Hàm refine query dựa trên lịch sử hội thoại
def refine_query(user_query, chat_history):
    """
    Refine câu hỏi của người dùng dựa trên lịch sử hội thoại, sử dụng prompt mới với phương pháp Chain of Thought (CoT).
    """
    try:
        logger.info(f"Query: {user_query}")
        llm = ChatGroq(
            model_name="deepseek-r1-distill-qwen-32b",
            temperature=0.7,
            api_key=api_key_llm
        )
        # Thực hiện refine câu hỏi
        refined_query_chain = (
            query_refinement_prompt_template
            | llm  # Sử dụng Large Language Model để sinh output
            | StrOutputParser()  # Parser để chuyển đổi output về dạng string
        )
        
        print(chat_history)
        refined_query = refined_query_chain.invoke({"chat_history": chat_history, "user_query": user_query})
    except Exception as e:
        logger.error(f"Lỗi khi refine query: {e}")
        # Trường hợp lỗi, giữ nguyên câu hỏi gốc
        refined_query = user_query
    
    return refined_query

if __name__ == "__main__":
    # Câu hỏi gốc
    api_key_llm = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model_name= "deepseek-r1-distill-qwen-32b",
        temperature= 0.7,
        api_key= api_key_llm
        )
    logger.info("Khởi tạo mô hình deepseek-r1 thành công.")
    chat_history = []

    ''' Test refine query '''
    # Refine câu hỏi
    while True:
        user_query = input("Nhập câu hỏi của bạn: ")	
        refined_query = refine_query(user_query, chat_history)
        logger.info(f"\nCâu hỏi sau refine: {refined_query}")

        # Thêm câu hỏi vào lịch sử hội thoại
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": refined_query})	
        logger.info(f"Lịch sử hội thoại: {chat_history}\n")