from langchain.prompts import PromptTemplate
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Query Prompts")

# Prompt 1: Rewrite Query
rewrite_query_prompt_template = """
Bạn là một trợ lý AI chuyên về du lịch. Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và xử lý như sau:

1. Kiểm tra xem câu hỏi có liên quan đến du lịch không (ví dụ: thời tiết, địa điểm, phương tiện di chuyển, chỗ ở, v.v.). Nếu không, trả về: "NON_TRAVEL: Câu hỏi của bạn không liên quan đến du lịch. Bạn có muốn hỏi gì về du lịch không?".
2. Nếu câu hỏi liên quan đến du lịch:
   - Xác định các yếu tố chính: chủ đề (ví dụ: thời tiết, chỗ ở), địa điểm, thời gian, đối tượng (nếu có).
   - Dựa trên ngữ cảnh để viết lại câu hỏi thành một phiên bản rõ ràng và dễ hiểu hơn (nếu cần).
   - Trả về "READY: [câu hỏi đã rõ ràng hoặc được viết lại]" nếu câu hỏi đủ thông tin.
   - Trả về "UNCLEAR: [câu hỏi gốc]" nếu thiếu thông tin và ngữ cảnh không đủ để làm rõ.

**Câu hỏi của người dùng:** {user_query}  
**Ngữ cảnh trước đó:** {chat_history}  

**Kết quả trả về:**
"""
rewrite_query_prompt = PromptTemplate(
   input_variables=["user_query", "chat_history"],
   template=rewrite_query_prompt_template
)
# Prompt 2: Clarify Query
clarify_query_prompt_template = """
Bạn là một trợ lý AI chuyên về du lịch. Nhiệm vụ của bạn là tạo câu hỏi làm rõ khi câu hỏi của người dùng không đủ thông tin hoặc không liên quan đến du lịch. Hãy làm như sau:

1. Nếu câu hỏi không liên quan đến du lịch, hỏi lại để hướng người dùng về chủ đề du lịch.
2. Nếu câu hỏi liên quan đến du lịch nhưng thiếu thông tin:
   - Xác định phần thông tin còn thiếu (ví dụ: địa điểm, thời gian, chủ đề cụ thể).
   - Dựa trên ngữ cảnh (nếu có) để đặt một câu hỏi làm rõ ngắn gọn, tự nhiên.
3. Trả về "ASK: [câu hỏi làm rõ]" dưới dạng ngắn gọn và thân thiện.

**Câu hỏi không rõ ràng:** {unclear_query}  
**Ngữ cảnh trước đó:** {chat_history}  

**Kết quả trả về:**
"""
clarify_query_prompt = PromptTemplate(
   input_variables=["unclear_query", "chat_history"],
   template=clarify_query_prompt_template
)