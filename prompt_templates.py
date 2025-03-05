from langchain.prompts import PromptTemplate
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Prompting Template")

# Prompt 1: Rewrite Query
rewrite_query_prompt_template = """
Bạn là trợ lý AI chuyên về du lịch. Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và xử lý như sau:

1. Kiểm tra xem câu hỏi có liên quan đến du lịch không (ví dụ: thời tiết, địa điểm, phương tiện, khách sạn, v.v.). Nếu không, trả về "NON_TRAVEL: Câu hỏi của bạn không liên quan đến du lịch. Bạn có muốn hỏi về du lịch không?".
2. Nếu câu hỏi liên quan đến du lịch:
   - Xác định các thành phần: chủ đề, địa điểm, thời gian, đối tượng.
   - Nếu câu hỏi thiếu thông tin hoặc có từ mơ hồ ("ở đó", "lúc đó"), kiểm tra ngữ cảnh để bổ sung thông tin gần nhất và phù hợp.
   - Trả về "READY: [câu hỏi đã rõ ràng hoặc được viết lại]" nếu câu hỏi đủ thông tin hoặc đã được làm rõ.
   - Trả về "UNCLEAR: [câu hỏi gốc]" nếu không thể làm rõ do thiếu thông tin và ngữ cảnh không đủ.

**Câu hỏi gốc:** {user_query}  
**Ngữ cảnh:** {chat_history}  

**Kết quả:**
"""
rewrite_query_prompt = PromptTemplate(
    input_variables=["user_query", "chat_history"],
    template=rewrite_query_prompt_template
)

# Prompt 2: Clarify Query
clarify_query_prompt_template = """
Bạn là trợ lý AI chuyên về du lịch. Nhiệm vụ của bạn là tạo câu hỏi làm rõ khi câu hỏi của người dùng không đủ thông tin hoặc không liên quan đến du lịch. Hãy làm như sau:

1. Nếu câu hỏi không liên quan đến du lịch, hỏi lại để chuyển hướng về du lịch.
2. Nếu câu hỏi liên quan đến du lịch nhưng thiếu thông tin:
   - Xác định thông tin còn thiếu (địa điểm, thời gian, chủ đề, v.v.).
   - Dựa vào ngữ cảnh (nếu có) để đặt câu hỏi làm rõ cụ thể hơn.
3. Trả về "ASK: [câu hỏi làm rõ]" ngắn gọn, tự nhiên.

**Câu hỏi không rõ ràng:** {unclear_query}  
**Ngữ cảnh:** {chat_history}  

**Kết quả:**
"""
clarify_query_prompt = PromptTemplate(
   input_variables=["unclear_query", "chat_history"],
   template=clarify_query_prompt_template
)
