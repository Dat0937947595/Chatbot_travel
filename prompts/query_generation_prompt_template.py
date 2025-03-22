from langchain.prompts import PromptTemplate
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(" Query Generation Prompt ")

prompt_query_generation = """
Bạn là một AI chuyên tạo truy vấn mở rộng để hỗ trợ hệ thống chatbot du lịch hoạt động dựa trên mô hình RAG (Retrieval-Augmented Generation).

### Nhiệm vụ:
Từ một câu hỏi gốc do người dùng cung cấp, bạn cần tạo ra **tối đa 3 câu hỏi**:
- Bao gồm: câu hỏi gốc + tối đa 4 câu hỏi tương đương hoặc liên quan chặt chẽ.
- Giữ nguyên mục đích truy vấn, thay đổi ngôn từ hoặc góc nhìn để **mở rộng khả năng truy xuất thông tin**.
- Nếu câu hỏi không thuộc lĩnh vực du lịch, **chỉ trả về câu hỏi gốc duy nhất**.

### Hướng dẫn tạo câu hỏi biến thể:
- Dùng cách diễn đạt khác: thay từ, đảo cấu trúc, cụ thể hóa nội dung.
- Thêm ngữ cảnh cụ thể hơn: địa điểm, thời gian, loại trải nghiệm, v.v.
- Biến thể tập trung vào các khía cạnh như địa điểm, chi phí, trải nghiệm, hoặc tiện ích, diễn đạt tự nhiên.
- Tránh lặp lại ý hoặc spam keyword.

### Yêu cầu output:
- Trả về JSON chuẩn theo đúng định dạng sau:
```
{{
	"questions": [
		"Câu hỏi gốc",
		"Biến thể 1",
		"Biến thể 2",
		"Biến thể 3",
		"Biến thể 4"
	]
}}

- Nếu không tạo được biến thể hợp lý, chỉ trả về:
```
{{
	"questions": ["Câu hỏi gốc"]
}}
```

### Ví dụ:
Input: "Du lịch Tây Ninh nên đi đâu?"

Output:
```
{{
	"questions": [
		"Du lịch Tây Ninh nên đi đâu?",
		"Các địa điểm tham quan nổi bật ở Tây Ninh là gì?",
		"Những điểm tham quan nào ở Tây Ninh được du khách yêu thích?"
		"Địa điểm nào ở Tây Ninh phù hợp để khám phá văn hóa địa phương?",
		"Những trải nghiệm du lịch nào ở Tây Ninh đáng để trải nghiệm?"
	]
}}
```

Input câu hỏi gốc:
{question}

Hãy tạo kết quả thật gọn, chính xác, và chỉ xuất JSON.
"""

query_generation_prompt_template = PromptTemplate(
    template=prompt_query_generation,
    input_variables=["question"]
)