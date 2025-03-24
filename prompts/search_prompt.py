from langchain.prompts import PromptTemplate
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Weather Info Prompt")


extract_search_info_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    Bạn là một trợ lý du lịch thông minh, chuyên nghiệp. Nhiệm vụ:
    1. Trích xuất thông tin từ câu hỏi của người dùng:
        - Nếu hỏi về giá vé khu vui chơi:
            + Tên khu vui chơi (amusement park name).
            + Thành phố (city).
        - Nếu hỏi về giá vé xe:
            + Thành phố đi (origin city).
            + Thành phố đến (destination city).
            + Ngày đi (departure date, mặc định là hôm nay nếu không rõ).
        - Nếu hỏi về giá vé tàu:
            + Thành phố đi (origin city).
            + Thành phố đến (destination city).
            + Ngày đi (departure date, mặc định là hôm nay nếu không rõ).
    2. Kiểm tra tên tỉnh/thành phố có hợp lệ không. Nếu không, sử dụng kiến thức của bạn về tỉnh thành phố Việt Nam để viết lại tên tỉnh/thành phố.
    3. Ngày được định dạng dưới dạng YYYY-MM-DD.

    **Câu hỏi người dùng**: "{query}"

    **Định dạng đầu ra (JSON)**:
    - Nếu hỏi về giá vé khu vui chơi: {{"type": "amusement_park", "park_name": "<tên khu vui chơi>", "city": "<thành phố>"}}
    - Nếu hỏi về giá vé xe: {{"type": "bus", "origin": "<thành phố đi>", "destination": "<thành phố đến>", "departure_date": "<ngày đi>"}}
    - Nếu hỏi về giá vé tàu: {{"type": "train", "origin": "<thành phố đi>", "destination": "<thành phố đến>", "departure_date": "<ngày đi>"}}
    - Nếu không rõ: {{"type": "unknown"}}
    """
)