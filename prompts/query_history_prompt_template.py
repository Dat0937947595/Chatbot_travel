from langchain.prompts import PromptTemplate
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Query History Prompt")

query_history_prompt = """
Bạn là một trợ lý du lịch thông minh, xử lý truy vấn dựa trên lịch sử hội thoại. Lịch sử là danh sách các lượt hội thoại: [{{"role": "user" hoặc "assistant", "content": "nội dung"}}]. Nhiệm vụ: Phân tích lịch sử, tinh chỉnh truy vấn nếu cần để tạo câu hỏi hoàn chỉnh.

---

### Hướng dẫn
1. **Phân tích**:
   - Xem `{history}` để tìm ngữ cảnh liên quan đến `{query}`.
   - Nếu cần, tinh chỉnh `{query}` thành câu hỏi hoàn chỉnh dựa trên ngữ cảnh.
   - Nếu `{query}` đã đầy đủ, giữ nguyên.

2. **Đầu ra** (JSON):
   - Trả về: `{{"refined_query": "câu hỏi hoàn chỉnh"}}`.

---

### Đầu vào
- **Lịch sử hội thoại**: {history}
- **Truy vấn**: {query}

---

### Ví dụ
1. **Lịch sử**: [{{"role": "user", "content": "Tôi muốn đi Đà Nẵng"}}, {{"role": "assistant", "content": "Bạn muốn đi Đà Nẵng khi nào?"}}]
   **Truy vấn**: "Thời tiết thế nào?"
   **Đầu ra**: {{"refined_query": "Thời tiết ở Đà Nẵng thế nào?"}}

2. **Lịch sử**: [{{"role": "user", "content": "Tôi thuê xe ở Đà Nẵng"}}, {{"role": "assistant", "content": "Bạn thuê xe gì và bao lâu?"}}]
   **Truy vấn**: "Toyota Vios 5 ngày, giá 500k/ngày"
   **Đầu ra**: {{"refined_query": "Tổng tiền thuê xe Toyota Vios ở Đà Nẵng trong 5 ngày với giá 500.000 VND/ngày là bao nhiêu?"}}

3. **Lịch sử**: [{{"role": "user", "content": "Các địa điểm du lịch ở Đà Nẵng?"}}, {{"role": "assistant", "content": "Có Bà Nà Hills, Ngũ Hành Sơn..."}}]
   **Truy vấn**: "Xa trung tâm không?"
   **Đầu ra**: {{"refined_query": "Bà Nà Hills và Ngũ Hành Sơn ở Đà Nẵng cách trung tâm bao xa?"}}

---

### Đầu ra cuối cùng
Trả về JSON: `{{"refined_query": "câu hỏi hoàn chỉnh"}}`.
"""

query_history_prompt_template = PromptTemplate(
   input_variables=["query", "history"],
   template=query_history_prompt
)

# Viết prompt viết câu hỏi hỏi người dùng thông tin còn thiếu:
missing_prompt = """
Bạn là một trợ lý du lịch thông minh. Nhiệm vụ của bạn là giúp người dùng hoàn thiện truy vấn du lịch bằng cách hỏi lại những thông tin còn thiếu.

---
# Đầu vào:
- Truy vấn của người dùng: {query}
- Các thông tin còn thiếu: {missing_info}

---
# Đầu ra:
Viết một câu hỏi tự nhiên và rõ ràng để yêu cầu người dùng cung cấp thông tin còn thiếu, sao cho truy vấn có thể được xử lý chính xác nhất.

Câu hỏi:
"""

missing_prompt_template = PromptTemplate(
   input_variables=["query", "missing_info"],
   template=missing_prompt
)