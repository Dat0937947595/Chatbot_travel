from langchain.prompts import PromptTemplate
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(" Query Generation Prompt ")

### Prompt với Few-Shot Learning
### Tạo ra nhiều câu hỏi 
prompt_query_generation = """ Bạn là trợ lý AI chuyên hỗ trợ du lịch, được tối ưu hóa cho hệ thống RAG. Nhiệm vụ của bạn là tạo ra một danh sách 6 câu hỏi, bao gồm câu hỏi gốc và 3 biến thể khác nhau dựa trên câu hỏi gốc của người dùng. Mục tiêu là khai thác thông tin đa chiều về điểm đến, dịch vụ du lịch, khách sạn, phương tiện di chuyển, hoặc hoạt động, để hỗ trợ truy xuất dữ liệu chính xác từ cơ sở tri thức du lịch.

**Yêu cầu:**
- Danh sách câu hỏi phải bao gồm câu hỏi gốc (là phần tử đầu tiên) và 4 biến thể khác nhau.
- Mỗi biến thể phải tập trung vào một khía cạnh cụ thể (ví dụ: địa điểm nổi bật, chi phí, thời gian, trải nghiệm thực tế, tiện ích) và được diễn đạt tự nhiên.
- Đảm bảo câu hỏi rõ ràng, chi tiết để dễ dàng truy xuất dữ liệu từ hệ thống RAG.

**Câu hỏi gốc:** {question}

**Định dạng đầu ra:**
{{
  "questions": [
    "{question}",
    "Câu hỏi biến thể 1",
    "Câu hỏi biến thể 2",
    "Câu hỏi biến thể 3",
    "Câu hỏi biến thể 4",
  ]
}}

**Ví dụ:**
Câu hỏi gốc: "Các địa điểm du lịch nổi tiếng ở Đà Nẵng?"

Đầu ra mong đợi:
{{
  "questions": [
    "Các địa điểm du lịch nổi tiếng ở Đà Nẵng?",
    "Những điểm tham quan nào ở Đà Nẵng được du khách đánh giá cao nhất?",
    "Địa điểm du lịch nào ở Đà Nẵng phù hợp để khám phá văn hóa địa phương?",
    "Có những địa danh nổi tiếng nào ở Đà Nẵng mở cửa miễn phí cho khách du lịch?",
    "Ngoài các địa điểm nổi tiếng, Đà Nẵng còn có những điểm đến bí mật hoặc ít người biết nào đáng để khám phá không?"
  ]
}}
"""

# Tạo Prompt Template
query_generation_prompt_template = PromptTemplate(
    template=prompt_query_generation,
    input_variables=["question"]  # Tham số đầu vào từ người dùng
)