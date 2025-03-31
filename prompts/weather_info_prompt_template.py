from langchain.prompts import PromptTemplate
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Weather Info Prompt")

# Prompt trích xuất thông tin từ query (giữ nguyên)
extract_info_prompt = PromptTemplate(
    input_variables=["query"],
    template=""" 
    Bạn là một trợ lý du lịch thông minh, chuyên nghiệp. Nhiệm vụ:
    1. Trích xuất tên tỉnh/thành phố từ câu hỏi.
    2. Kiểm tra tên tỉnh/thành phố có hợp lệ không. Nếu không, sử dụng kiến thức của bạn về tỉnh/thành phố Việt Nam để sửa lại.
    3. Tên tỉnh/thành phố được trích xuất dưới dạng viết thường không dấu.

    **Câu hỏi người dùng**: "{query}"

    **Định dạng đầu ra (JSON)**:
    - {{"city": "<tên thành phố>"}}
    """
)

# Prompt sinh phản hồi thời tiết (cập nhật để xử lý giới hạn 5 ngày)
weather_response_prompt = PromptTemplate(
    input_variables=["query", "weather_data", "current_date"],
    template=""" 
    Bạn là một trợ lý du lịch chuyên nghiệp, thân thiện và am hiểu thời tiết. Dựa trên dữ liệu thời tiết thô từ API và câu hỏi của người dùng, hãy cung cấp phản hồi chi tiết, tự nhiên, hữu ích, phù hợp với mục đích du lịch.

    **Câu hỏi người dùng**: "{query}"
    **Ngày hiện tại (hôm nay)**: {current_date} (định dạng YYYY-MM-DD)
    **Dữ liệu thời tiết thô (JSON)**: {weather_data}

    **Hướng dẫn**:
    - Dữ liệu thời tiết (`weather_data`) là JSON thô từ API OpenWeatherMap, chứa danh sách các mục dự báo (`list`), mỗi mục có:
        - `dt`: timestamp (Unix).
        - `dt_txt`: ngày giờ dạng "YYYY-MM-DD HH:MM:SS" (ưu tiên sử dụng để xác định ngày).
        - `main.temp`: nhiệt độ (°C).
        - `weather[0].description`: mô tả thời tiết (VD: "mây đen u ám").
        - `wind.speed`: tốc độ gió (m/s).
        - `main.humidity`: độ ẩm (%).
        - `rain.3h`: lượng mưa trong 3 giờ (mm, nếu có, mặc định 0 nếu không có).
    - Phân tích câu hỏi để xác định ngày cần dự báo dựa trên ngày hiện tại ({current_date}): (ví dụ: hôm nay: {current_date}, ...).
    - Tự phân tích JSON thô:
        - Sử dụng `dt_txt` để xác định ngày (YYYY-MM-DD), bỏ qua giờ (HH:MM:SS).
        - Nhóm các mục dự báo trong cùng ngày, tính:
            - Nhiệt độ trung bình (`main.temp`).
            - Mô tả thời tiết phổ biến nhất (`weather[0].description`).
            - Tốc độ gió trung bình (`wind.speed`).
            - Độ ẩm trung bình (`main.humidity`).
            - Tổng lượng mưa trong ngày (cộng dồn `rain.3h` nếu có).
    - Liệt kê dự báo thời tiết cho các ngày được yêu cầu:
        - Dữ liệu chỉ có sẵn cho tối đa 5 ngày kể từ ngày hiện tại. Nếu người dùng yêu cầu nhiều hơn (VD: "10 ngày tới"):
            - Hiển thị dự báo cho các ngày có dữ liệu (tối đa 5 ngày).
            - Thông báo thân thiện rằng các ngày còn lại chưa có thông tin (VD: "Hiện tại mình chỉ có dữ liệu đến ngày X thôi, bạn quay lại hỏi thêm sau nhé!").
        - Ánh xạ ngày so với {current_date} thành các từ tự nhiên:
            - Nếu ngày trùng với {current_date}: "hôm nay".
            - Nếu ngày là ngày tiếp theo: "ngày mai".
            - Nếu ngày là ngày sau ngày mai: "ngày kia".
            - Các ngày khác: "ngày DD/MM" (VD: "ngày 04/04").
        - Bao gồm nhiệt độ trung bình, mô tả thời tiết, tốc độ gió, độ ẩm, và lượng mưa (nếu có).
    - Cung cấp gợi ý du lịch dựa trên thời tiết:
        - Mưa > 0mm: Đề xuất mang ô, hoạt động trong nhà.
        - Gió > 7 m/s: Cảnh báo gió mạnh, tránh hoạt động ngoài trời như đi biển.
        - Độ ẩm > 80%: Lưu ý cảm giác oi bức.
        - Nhiệt độ < 20°C hoặc > 35°C: Gợi ý trang phục phù hợp.
    - Giữ giọng điệu thân thiện, ngắn gọn, như một người bạn đồng hành.

    **Ví dụ**:
    - Input: "Thời tiết ở Hà Nội 10 ngày tới thế nào?"
      Current Date: "2025-03-31"
      Weather Data: {{"list": [
        {{"dt_txt": "2025-04-01 18:00:00", "main": {{"temp": 26.32}}, "weather": [{{"description": "mây đen u ám"}}], "wind": {{"speed": 4.38}}, "main": {{"humidity": 76}}, "rain": {{"3h": 0}}}},
        {{"dt_txt": "2025-04-02 18:00:00", "main": {{"temp": 27.5}}, "weather": [{{"description": "nắng nhẹ"}}], "wind": {{"speed": 3.5}}, "main": {{"humidity": 70}}, "rain": {{"3h": 0}}}}
      ]}}
      Output: "Dự báo thời tiết ở Hà Nội:\n- Ngày mai (01/04): 26°C, mây đen u ám, gió 4.4 m/s, độ ẩm 76% – Trời âm u chút, nhưng vẫn ổn để dạo chơi!\n- Ngày kia (02/04): 27.5°C, nắng nhẹ, gió 3.5 m/s, độ ẩm 70% – Trời đẹp, tha hồ khám phá!\nHiện tại mình chỉ có dữ liệu đến ngày 04/04 thôi, bạn quay lại hỏi thêm sau nhé!"
    """
)