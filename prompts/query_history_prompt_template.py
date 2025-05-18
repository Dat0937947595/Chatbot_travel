from langchain.prompts import PromptTemplate
import logging
import datetime
import time

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Query History Prompt")

query_history_prompt = """
Bạn là một trợ lý du lịch thông minh, xử lý truy vấn dựa trên lịch sử hội thoại. Lịch sử là danh sách các lượt hội thoại. 

Nhiệm vụ: Phân tích lịch sử, tinh chỉnh truy vấn nếu cần để tạo câu hỏi hoàn chỉnh phục vụ cho việc tìm kiếm thông tin, trả lời câu hỏi dễ dàng hơn.

Lưu ý: Câu hỏi được viết lại thành một câu hỏi hoàn chỉnh, rõ ràng và dễ hiểu hơn.

---  
Định nghĩa “câu hỏi hoàn chỉnh”:  
- Đưa vào thông tin ngữ cảnh quan trọng từ lịch sử (địa điểm, thời gian, mục đích, đối tượng…)  
- Viết thành câu hỏi đầy đủ, không thiếu ngữ pháp, rõ ràng. 

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
- **Thời gian hiện tại:** {current_time} từ thời gian hiện tại bạn có thể tự suy ra thời gian của ngày hôm qua, tuần trước, tháng trước, năm trước, ... để đưa vào câu hỏi hoàn chỉnh.
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

### Đầu ra cuối cùng:
Trả về JSON: 
```json
{{"refined_query": "câu hỏi hoàn chỉnh"}}
```.
"""

query_history_prompt_template = PromptTemplate(
   input_variables=["query", "history", "current_time"],
   template=query_history_prompt
).partial(
   current_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)