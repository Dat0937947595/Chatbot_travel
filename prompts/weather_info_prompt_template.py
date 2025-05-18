from langchain.prompts import PromptTemplate
import time
import datetime

# 1) Prompt trích xuất city
extract_info_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    Bạn là trợ lý du lịch siêu xịn. 
    Nhiệm vụ:
        - Lấy tên tỉnh/thành phố từ câu hỏi.
        - Nếu sai hoặc không hợp lệ, sửa lại dựa trên kiến thức địa danh Việt Nam.
        - Trả về lower-case, không dấu.
        - Tính toán ngày người dùng mong muốn dự báo thời tiết (nếu có). Nếu không có mặc đinh là ngày hiện tại.

    User query: "{query}"
    Output JSON: {{"city": "<tên thành phố>"}}
    Ngày hiện tại: {current_date}
    """
).partial(current_date=datetime.datetime.now().strftime("%Y-%m-%d"))

# 2) Prompt phản hồi thời tiết
# Prompt sinh phản hồi thời tiết tối ưu, kết hợp Pydantic và xử lý tình huống
weather_response_prompt = PromptTemplate(
    input_variables=["query", "weather_data", "current_date", "requested_dates", "amount"],
    template="""
Bạn là trợ lý du lịch thân thiện và am hiểu thời tiết.

Thông tin đầu vào:
- Query gốc: "{query}"
- Ngày hiện tại: {current_date} (YYYY-MM-DD)
- Danh sách ngày người dùng yêu cầu (YYYY-MM-DD): {requested_dates}
- Dữ liệu raw JSON OpenWeather (forecast list)
{weather_data}

Hướng dẫn xử lý:
1. Trích xuất tất cả bản ghi dự báo từ JSON, nhóm theo `ngày`.
2. Với mỗi ngày trong {requested_dates}:
    - Nếu số lượng ngày {amount} > 5 ngày: chỉ dùng 5 ngày đầu tiên, và sau khi liệt kê, phản hồi thân thiện ví dụ: "Mình chỉ có dữ liệu tối đa 5 ngày thôi nhé!"
    - Nếu không có dữ liệu hoặc ngày đề cập trong quá khứ: phản hồi thân thiện ví dụ: "Xin lỗi, mình không có thông tin thời tiết cho ngày này."
    - Nếu có dữ liệu:
        • Tính trung bình nhiệt độ (°C), gió (m/s), độ ẩm (%), tổng mưa (mm).
        • Xác định mô tả thời tiết phổ biến nhất.
        • Gán nhãn:
        - current_date → "Hôm nay"
        - current_date +1 → "Ngày mai"
        - current_date +2 → "Ngày kia"
        - khác → "Ngày DD/MM"
        • Gợi ý tips:
        - Nếu tổng mưa > 0: "Mang ô dù nếu đi ra ngoài."
        - Nếu tốc độ gió trung bình > 7 m/s: "Lưu ý gió mạnh."
        - Nếu độ ẩm trung bình > 80%: "Không khí oi bức, cân nhắc đồ thoáng mát."
        - Nếu nhiệt độ trung bình < 20: "Trời se lạnh, nên mặc áo khoác nhẹ."
        - Nếu nhiệt độ trung bình > 35: "Rất nóng, trang phục thoáng mát và dưỡng ẩm đầy đủ."
3. Trình bày:
    - Dùng gạch đầu dòng cho mỗi ngày.
    - Bao gồm: Nhãn ngày, nhiệt độ, mô tả, gió, độ ẩm, mưa, tips, ...

Tone: Bạn là một trợ lý thân thiện.
"""
).partial(
    current_date=datetime.datetime.now().strftime("%Y-%m-%d")
)

