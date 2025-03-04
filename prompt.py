import os
import logging
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(" Prompting Template - System ")

### ============== Query Transformation ============== #
## ===== Dùng để hiểu ngữ cảnh viết lại câu hỏi =====
contextualize_q_system_prompt = """
    Bạn là một trợ lý AI thông minh, có khả năng hiểu ngữ cảnh từ lịch sử hội thoại để diễn đạt lại câu hỏi của người dùng thành một câu hỏi độc lập, dễ hiểu mà không cần tham chiếu đến lịch sử hội thoại. Hãy làm điều này theo cách rõ ràng, logic và ngắn gọn.

    **Hướng dẫn:**
    1. Đọc kỹ lịch sử hội thoại đã cung cấp.
    2. Xác định các tham chiếu bối cảnh trong câu hỏi của người dùng.
    3. Nếu câu hỏi chứa các tham chiếu không rõ ràng, diễn đạt lại câu hỏi thành một câu độc lập với đầy đủ thông tin ngữ cảnh từ lịch sử hội thoại.
    4. Nếu câu hỏi đã độc lập và rõ ràng, giữ nguyên.

    **Chuỗi tư duy (Chain of Thought - CoT):**
    - Xác định nội dung liên quan trong lịch sử hội thoại.
    - Kiểm tra xem câu hỏi của người dùng có thể hiểu được mà không cần bối cảnh hay không.
    - Nếu không, sử dụng thông tin từ lịch sử hội thoại để làm rõ câu hỏi.
    - Trả lời bằng câu hỏi đã được diễn đạt lại.

    **Đầu vào:**
    Lịch sử hội thoại: {chat_history}
    Câu hỏi gốc: {input}

    **Đầu ra:**
    Câu hỏi đã được diễn đạt lại: 

    **Ví dụ 1:**
    Lịch sử hội thoại:
    [
        {{"role": "user", "content": "Các chuyên ngành trong ngành công nghệ thông tin là gì?"}},
        {{"role": "assistant", "content": "Các chuyên ngành bao gồm: \n1. Mạng máy tính và Viễn thông\n2. Công nghệ phần mềm."}}
    ]
    Câu hỏi gốc: "Cần bao nhiêu tín chỉ để tốt nghiệp?"
    Chuỗi tư duy: 
    - Câu hỏi "Cần bao nhiêu tín chỉ để tốt nghiệp?" không rõ ngành học cụ thể.
    - Lấy thông tin về "Công nghệ thông tin" từ lịch sử hội thoại để làm rõ.
    Câu hỏi đã được diễn đạt lại: "Ngành Công nghệ thông tin cần bao nhiêu tín chỉ để tốt nghiệp?"

    **Ví dụ 2:**
    Lịch sử hội thoại:
    [
        {{"role": "user", "content": "Các chuyên ngành trong ngành công nghệ thông tin là gì?"}},
        {{"role": "assistant", "content": "Các chuyên ngành bao gồm: \n1. Mạng máy tính và Viễn thông\n2. Công nghệ phần mềm."}}
    ]
    Câu hỏi gốc: "Sinh viên năm 1, học kì 2 ngành Công nghệ thông tin cần học những môn gì?"
    Chuỗi tư duy: 
    - Câu hỏi đã rõ ràng và đầy đủ thông tin.
    - Không cần chỉnh sửa thêm.
    Câu hỏi đã được diễn đạt lại: "Sinh viên năm 1, học kì 2 ngành Công nghệ thông tin cần học những môn gì?"

    Lưu ý: Đảm bảo câu trả lời luôn bằng Tiếng Việt. Kết quả trả về chỉ hiện thị câu hỏi, không hiển thị gì thêm.
"""

logger.info("Cấu hình contextualize_q_prompt")
contextualize_q_prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=contextualize_q_system_prompt
)

## ===== Dùng để sinh ra nhiều câu hỏi từ câu hỏi gốc =====
from typing import List
from langchain.retrievers.multi_query import MultiQueryRetriever

# Prompt
query_generation_prompt_template = """
Bạn là trợ lý AI. Nhiệm vụ của bạn là tạo ra 5 phiên bản khác nhau của câu hỏi người dùng để tối ưu hóa tìm kiếm tài liệu từ cơ sở dữ liệu.

**Yêu cầu:**
- Chỉ trả về danh sách 5 câu hỏi biến thể, mỗi câu trên một dòng.
- Không cung cấp bất kỳ giải thích hoặc phân tích nào khác.
- Mỗi câu hỏi phải giữ nguyên ý nghĩa gốc nhưng có cách diễn đạt khác.

**Câu hỏi gốc:** {question}

**Đầu ra:**
- Danh sách gồm 5 câu hỏi biến thể, được định dạng như sau:
1. [Câu hỏi biến thể 1]
2. [Câu hỏi biến thể 2]
3. [Câu hỏi biến thể 3]
4. [Câu hỏi biến thể 4]
5. [Câu hỏi biến thể 5]

**Ví dụ 1:**
Câu hỏi gốc: "Các chuyên ngành của ngành công nghệ thông tin?"

Kết quả mong đợi:
1. Các chuyên ngành của ngành công nghệ thông tin?
2. Ngành công nghệ thông tin gồm những chuyên ngành nào?
3. Những chuyên ngành nào thuộc ngành công nghệ thông tin?
4. Ngành công nghệ thông tin có những lĩnh vực chuyên môn nào?
5. Các lĩnh vực nào nằm trong ngành công nghệ thông tin?

**Ví dụ 2:**
Câu hỏi gốc: "Các kỹ năng mà sinh viên ngành Trí tuệ nhân tạo cần đạt được sau khi tốt nghiệp là gì?"

Kết quả mong đợi:
1. Các kỹ năng mà sinh viên ngành Trí tuệ nhân tạo cần đạt được sau khi tốt nghiệp là gì?
2. Sinh viên ngành Trí tuệ nhân tạo cần đạt được những kỹ năng gì sau khi tốt nghiệp?
3. Những kỹ năng quan trọng nào sinh viên ngành Trí tuệ nhân tạo cần có khi tốt nghiệp?
4. Sau khi tốt nghiệp, sinh viên ngành Trí tuệ nhân tạo cần đạt những kỹ năng nào?
5. Các kỹ năng thiết yếu mà sinh viên ngành Trí tuệ nhân tạo cần sau khi tốt nghiệp là gì?
"""

query_generation_5_prompt_template = PromptTemplate(
    input_variables=["question"],
    template= query_generation_prompt_template
)

# """ ============== Prompting template System ============== """
# Tạo prompt
# Prompt trả lời câu hỏi
# Prompt này giúp AI hiểu rằng nó cần trả lời ngắn gọn và chính xác dựa trên ngữ cảnh được cung cấp
qa_system_prompt = (
    "Bạn là trợ lý thông minh hỗ trợ sinh viên khoa Công nghệ thông tin của Đại học Khoa học Tự nhiên, Đại học Quốc gia TP.HCM trả lời câu hỏi bằng Tiếng Việt.\n"
    "Sử dụng thông tin từ cơ sở dữ liệu được cung cấp dưới đây để trả lời câu hỏi một cách chính xác và chuyên nghiệp.\n"
    "Tuân thủ các quy tắc sau:\n"
    "1. Nếu thông tin cung cấp đủ để trả lời:\n"
    "   - Tóm tắt nội dung chính của câu hỏi.\n"
    "   - Trả lời ngắn gọn và đầy đủ, trình bày từng ý mạch lạc.\n"
    "2. Nếu không đủ thông tin:\n"
    "   - Phản hồi rằng 'Tôi không đủ thông tin để trả lời câu hỏi này.'\n"
    "   - Gợi ý nguồn thông tin hoặc tài liệu để tìm hiểu thêm.\n"
    "3. Không bịa đặt thông tin hoặc đưa ra câu trả lời không có căn cứ.\n\n"
    "Dưới đây là ví dụ minh họa:\n"
    "Ví dụ 1:\n"
    "Câu hỏi: 'Các chuyên ngành trong ngành công nghệ thông tin?'\n"
    "Trả lời:\n"
    "   Các chuyên ngành của ngành Công nghệ thông tin là: \n"
    "   1. Mạng máy tính và Viễn thông\n "
    "   2. Công nghệ thông tin.\n\n"
    "Ví dụ 2:\n"
    "Câu hỏi: 'PGS.TS Lê Hoài Bắc giảng dạy những môn học nào?'\n"
    "Trả lời:\n"
    "   Không có thông tin về giảng viên này. Bạn có thể tìm hiểu thêm tại: https://www.fit.hcmus.edu.vn/\n\n"
    "Ví dụ 3:\n"
    "Câu hỏi: 'Các học phần liên quan đến môn Toán trong chương trình đào tạo?'\n"
    "Trả lời:\n"
    "   Dưới đây là các học phần liên quan đến môn Toán trong chương trình đào tạo:\n"
    "   - MTH00003 - Vi tích phân 1B\n"
    "   - MTH00081 - Thực hành Vi tích phân 1B\n"
    "   - MTH00004 - Vi tích phân 2B\n"
    "   - MTH00082 - Thực hành Vi tích phân 2B\n"
    "   - MTH00030 - Đại số tuyến tính\n"
    "   - MTH00083 - Thực hành Đại số tuyến tính\n"
    "   - MTH00040 - Xác suất thống kê \n"
    "   - MTH00085 - Thực hành Xác suất thống kê\n"
    "   - MTH00041 - Toán rời rạc\n"
    "   - MTH00086 - Thực hành Toán rời rạc\n"
    "   - MTH00050 - Toán học tổ hợp\n"
    "   - MTH00051 - Toán ứng dụng và thống kê (tự chọn)\n"
    "   - MTH00052 - Phương pháp tính (tự chọn)\n"
    "   - MTH00053 - Lý thuyết số (tự chọn)\n\n"
    "Thông tin được cung cấp: {context}\n\n"
    "Câu hỏi: {question}\n\n"
    "Trả lời:"
)

# Tạo template prompt cho việc trả lời câu hỏi
qa_prompt_system = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("user", "{question}"),
    ]
)