from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
from pydantic import Field
import re

os.environ["GOOGLE_API_KEY"] = "AIzaSyCxKATX40xEOqEZCAtusTIdlXk7Z9C74KE"
os.environ["GOOGLE_CSE_ID"] = "033cd3a8777fa4c8d"

memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key="AIzaSyA-MAlE62P8Gg2g664zwnYcRAtNykEg_tE"
        )

price_prompt_template = PromptTemplate.from_template(
"""
Bạn là một hướng dẫn viên du lịch thông minh và thân thiện. Nhiệm vụ của bạn là cung cấp thông tin về giá cả các dịch vụ du lịch, hỗ trợ thông tin đặt vé các phương tiện di chuyển đồng thời tư vấn cho khách hàng như một chuyên gia, giúp họ đưa ra quyết định phù hợp nhất. Hãy đảm bảo rằng câu trả lời của bạn đầy đủ, tự nhiên và hữu ích.

### **Thông tin hội thoại trước đó**:
{chat_history}

### **Câu hỏi từ người dùng**:
{input}

### **Công cụ có sẵn**:
{tools}

### **Tên công cụ**:
{tool_names}

## **Quy tắc bắt buộc**:
1. Chỉ sử dụng thông tin từ công cụ để trả lời, không tự suy đoán.
2. Nếu chưa có đủ thông tin, sử dụng công cụ để tìm kiếm giá.
3. Khi đã có đủ thông tin, hãy tạo một câu trả lời hoàn chỉnh, không cần tìm kiếm thêm.
4. Luôn cung cấp một câu trả lời chi tiết như một hướng dẫn viên du lịch, không chỉ đưa ra con số mà còn giúp người dùng hiểu rõ hơn về dịch vụ.
5. Tích hợp thông tin giá vào câu trả lời theo cách tự nhiên, không chỉ liệt kê mà còn phân tích ưu và nhược điểm nếu có.
6. Luôn tham khảo ít nhất 10 trang web để đảm bảo thông tin cập nhật và chính xác.
7. Cung cấp danh sách các đường link tham khảo ở cuối câu trả lời.

---

## **Hướng dẫn cách phản hồi**:

- Nếu answer từ công cụ chứa giá vé hoặc giá dịch vụ, hãy diễn giải thông tin đó một cách tự nhiên, thêm lời khuyên hữu ích và đề xuất các lựa chọn phù hợp cho người dùng.
- Nếu answer có nhiều mức giá khác nhau, hãy so sánh và gợi ý phương án tốt nhất tùy theo từng trường hợp.
- Nếu answer không chứa thông tin cần thiết, hãy tiếp tục tìm kiếm để đảm bảo cung cấp đủ thông tin.
- Lấy các liên kết tham khảo ở references

---

## **Định dạng bắt buộc**:

Thought: Mô tả suy nghĩ của bạn về câu hỏi.
Action: <Tên công cụ> (chỉ khi cần tìm kiếm thêm thông tin)
Action Input: <Thông tin cần tìm kiếm, bao gồm loại dịch vụ, địa điểm, ngày cụ thể (nếu có), nếu không có ngày cụ thể hãy tự thêm từ "mới nhất">
Observation: <Thông tin thu được từ công cụ>. Tham khảo: <Danh sách các liên kết tham khảo ở references>

Final Answer: <Kết quả xuất ra>

**Gợi ý thêm cho người dùng**:
- Nếu có nhiều mức giá, hãy so sánh và đưa ra lựa chọn phù hợp.
- Nếu có chương trình khuyến mãi hoặc ưu đãi đặc biệt, hãy đề cập.
- Nếu có nhiều phương án thay thế (ví dụ: xe khách, xe limousine, tàu hỏa), hãy giới thiệu một cách tự nhiên.

Bạn có thể kiểm tra thông tin chi tiết và đặt vé tại: <Danh sách các liên kết tham khảo>


---

## **Ví dụ 1: Tư vấn giá vé xe khách từ Cần Thơ đi Sài Gòn**

Thought: Người dùng muốn biết giá vé xe khách từ Cần Thơ đi Sài Gòn. Tôi sẽ tìm thông tin mới nhất.
Action: Price_Search
Action Input: Giá vé xe khách Cần Thơ - Sài Gòn mới nhất
Observation: Giá vé xe khách dao động từ 150.000 VNĐ đến 350.000 VNĐ tùy vào loại xe và hãng xe. Các hãng phổ biến gồm Phương Trang, Thành Bưởi, và Kumho Samco.  
Tham khảo: [Link 1], [Link 2], [Link 3]

Final Answer: **Giá vé xe khách từ Cần Thơ đi Sài Gòn** dao động từ **150.000 VNĐ - 350.000 VNĐ** tùy vào hãng xe và loại xe.  
- **Phương Trang**: Khoảng 180.000 VNĐ, xe giường nằm, chất lượng cao.  
- **Thành Bưởi**: Khoảng 200.000 VNĐ, xe limousine, tiện nghi hơn.  
- **Kumho Samco**: Khoảng 150.000 VNĐ, giá rẻ hơn nhưng ít chuyến hơn.  

**Lời khuyên từ hướng dẫn viên**:  
- Nếu bạn muốn trải nghiệm thoải mái hơn, hãy chọn xe **limousine** của Thành Bưởi.  
- Nếu ưu tiên giá rẻ, Kumho Samco là một lựa chọn tốt.  
- Hãy đặt vé trước vào cuối tuần hoặc dịp lễ để tránh hết chỗ.  

Bạn có thể kiểm tra thông tin chi tiết và đặt vé tại:  
[Link 1]  
[Link 2]  
[Link 3]  

---

## **Ví dụ 2: Tư vấn giá vé tham quan Bà Nà Hills**

Thought: Người dùng muốn biết giá vé tham quan Bà Nà Hills mới nhất.
Action: Price_Search
Action Input: "Giá vé Bà Nà Hills mới nhất"
Observation: Giá vé người lớn: 850.000 VNĐ, trẻ em (1m-1m4): 700.000 VNĐ. Đã bao gồm vé cáp treo, tham quan, buffet trưa.  
Tham khảo: [Link 4], [Link 5]

Final Answer: **Giá vé tham quan Bà Nà Hills** hiện tại:  
- **Người lớn**: **850.000 VNĐ**  
- **Trẻ em (1m-1m4)**: **700.000 VNĐ**  
**Lưu ý**: Giá vé đã bao gồm vé cáp treo, tham quan các điểm nổi bật như Cầu Vàng, Làng Pháp, và buffet trưa.  

**Gợi ý từ hướng dẫn viên**:  
- Nếu bạn đi vào sáng sớm, hãy chuẩn bị áo khoác vì thời tiết có thể hơi lạnh.  
- Nên đặt vé trước để tránh xếp hàng lâu.  
- Nếu muốn trải nghiệm ẩm thực, buffet trong vé là lựa chọn tốt nhưng có thể đặt nhà hàng riêng để có trải nghiệm cao cấp hơn.  

Bạn có thể xem thông tin chi tiết và đặt vé tại:  
[Link 4]  
[Link 5]  

---

### **Ví dụ 3: Tư vấn về cách đặt xe đi TP. Hồ Chí Minh**  

Thought: Người dùng muốn biết cách đặt xe đi TP. Hồ Chí Minh. Tôi cần tìm thông tin về các dịch vụ đặt xe mới nhất.  
Action: Link_Search  
Action Input: "Cách đặt xe đi TP. Hồ Chí Minh mới nhất"  
Observation: Hiện có nhiều phương thức đặt xe đi TP. Hồ Chí Minh, bao gồm xe khách, xe limousine, taxi, và xe hợp đồng. Một số nhà xe phổ biến như **Phương Trang, Thành Bưởi, Mai Linh, Vinasun** đều có dịch vụ đặt xe trực tuyến.  
Tham khảo: [Link 1], [Link 2], [Link 3], [Link 4]  

Final Answer:  
**Cách đặt xe đi TP. Hồ Chí Minh** có nhiều phương án tùy theo nhu cầu của bạn:  

### 1. **Xe khách giường nằm**  
- **Phương Trang**: Xuất bến liên tục, đặt vé qua app hoặc website.  
- **Thành Bưởi**: Xe chất lượng cao, có ghế ngả và giường nằm.  
- **Giá vé**: Dao động từ **180.000 VNĐ - 350.000 VNĐ** tùy loại xe và tuyến đường.  

Đặt xe ngay:  
[Phương Trang](Link 1) | [Thành Bưởi](Link 2)  

---

### 2. **Xe limousine cao cấp**  
- **Xe limousine 9-16 chỗ**: Tiện nghi hơn, có ghế massage, nước uống.  
- **Giá vé**: Từ **250.000 VNĐ - 500.000 VNĐ**.  

Đặt limousine tại:  
[Vexere](Link 3) | [Limody](Link 4)  

---

### 3. **Taxi hoặc xe hợp đồng**  
- **Taxi Mai Linh, Vinasun**: Đặt qua app, tổng đài hoặc vẫy xe trực tiếp.  
- **Xe hợp đồng**: Phù hợp nếu đi nhóm đông, có thể thuê xe riêng.  

Đặt xe tại:  
[Mai Linh](Link 5) | [Vinasun](Link 6)  

---

**Gợi ý từ hướng dẫn viên**:  
Nếu đi xa và cần tiết kiệm, **xe khách giường nằm** là lựa chọn tốt.  
Nếu cần tiện nghi và nhanh hơn, chọn **xe limousine**.  
Nếu đi theo nhóm hoặc cần sự linh hoạt, có thể đặt **xe hợp đồng**.  

Bạn có thể kiểm tra thông tin chi tiết và đặt xe tại:  
[Link 1] [Link 2] [Link 3] [Link 4] [Link 5] [Link 6] 

{agent_scratchpad}
STOP
"""
)


class CustomGoogleSearchAPIWrapperPrice(GoogleSearchAPIWrapper):
    use_selenium: bool = Field(default=False, exclude=True)  # Khai báo thuộc tính use_selenium

    def __init__(self, use_selenium=False, google_api_key=None, google_cse_id=None, **kwargs):
        """Khởi tạo với API Key và CSE ID có thể truyền từ tham số hoặc lấy từ mặc định."""
        google_api_key = google_api_key or "AIzaSyCxKATX40xEOqEZCAtusTIdlXk7Z9C74KE"
        google_cse_id = google_cse_id or "033cd3a8777fa4c8d"
        
        # Thêm API vào kwargs nếu chưa có
        kwargs.setdefault("google_api_key", google_api_key)
        kwargs.setdefault("google_cse_id", google_cse_id)

        super().__init__(**kwargs)  # Gọi constructor của lớp cha với kwargs
        object.__setattr__(self, "use_selenium", use_selenium)  # Đặt giá trị cho use_selenium

    def extract_price_from_html(self, html):
        """Trích xuất giá từ nội dung HTML bằng BeautifulSoup + regex."""
        soup = BeautifulSoup(html, "html.parser")
        price_candidates = []

        # Regex tìm giá tiền (VNĐ, USD, $)
        price_regex = re.compile(r"(\d{1,3}(?:[.,]\d{3})*(?:\s*(?:đ|VNĐ|\$|USD)))", re.IGNORECASE)

        # Tìm tất cả phần tử có thể chứa giá tiền
        for tag in soup.find_all(["span", "div", "p", "td"]):
            text = tag.get_text(strip=True)
            matches = price_regex.findall(text)
            if matches:
                price_candidates.extend(matches)

        return list(set(price_candidates))[:5]  #  Trả về tối đa 5 giá trị (loại bỏ trùng lặp)

    def get_page_content(self, url):
        """Truy cập trang web và lấy nội dung (dùng requests hoặc Selenium nếu cần)."""
        print(f"🔍 Đang lấy nội dung từ: {url}")  # ✅ Debug log

        if self.use_selenium:
            try:
                options = Options()
                options.add_argument("--headless")  # Chạy không hiển thị trình duyệt
                options.add_argument("--disable-blink-features=AutomationControlled")  # Giả lập trình duyệt người dùng
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                
                driver.get(url)
                time.sleep(5)  # Đợi trang tải xong hoàn toàn

                html = driver.page_source
                driver.quit()
                return html
            except Exception as e:
                print(f"Lỗi Selenium: {e}")
                return None
        else:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9"
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()  # Kiểm tra lỗi HTTP
                return response.text
            except requests.exceptions.RequestException as e:
                print(f"Lỗi Requests: {e}")
                return None

    def run(self, query):
        print(f"🔎 Đang tìm kiếm: {query}") 

        search_results = self.results(query, num_results=5)  # Lấy 5 kết quả đầu tiên
        if not search_results:
            print("Không tìm thấy kết quả nào!")
            return {"content": "Không tìm thấy kết quả phù hợp.", "references": []}

        content = ""
        references = []

        for result in search_results:
            url = result["link"]
            references.append(url)

            html = self.get_page_content(url)
            if html:
                extracted_prices = self.extract_price_from_html(html)
                if extracted_prices:
                    content += f"{result['title']}: {', '.join(extracted_prices)}\n"
                else:
                    content += f"{result['title']}: Không tìm thấy giá trực tiếp.\n"
            else:
                content += f"{result['title']}: Không thể lấy dữ liệu.\n"

        return {"answer": content, "references": references}
    
class CustomGoogleSearchAPIWrapperLink(GoogleSearchAPIWrapper):
    def run(self, query):
        search_results = self.results(query, num_results=10)
        content = ""
        references = []

        for result in search_results:
            # Lấy nội dung và liên kết của từng kết quả tìm kiếm
            content += f"{result['title']}: {result['snippet']}\n"
            references.append(result['link'])

        # Trả về cả nội dung và danh sách liên kết tham khảo
        return {"content": content, "references": references}

search_link = CustomGoogleSearchAPIWrapperLink()

# Khởi tạo công cụ tìm kiếm với Selenium để đọc trang JavaScript
search_price = CustomGoogleSearchAPIWrapperPrice(use_selenium=True)# Tạo công cụ tìm kiếm giá


price_search_tool = Tool(
  name="Price_Search",
  func=search_price.run,
  description="Tìm kiếm thông tin về giá của những dịch vụ của du lịch mới nhất trên Google và trả về cả liên kết tham khảo"
)

# Tạo công cụ tìm kiếm
link_search_tool = Tool(
    name="Link_Search",
    func=search_link.run,
    description="Tìm kiếm thông tin về những trang web cung cấp dịch vụ về du lịch mà người dùng cần"
)



# Khởi tạo React Agent
price_link_react_agent = create_react_agent(
  llm=llm_gemini,
  tools=[price_search_tool, link_search_tool],
  prompt=price_prompt_template
)

# Tạo AgentExecutor
agent_price_executor = AgentExecutor(
  agent=price_link_react_agent,
  tools=[price_search_tool, link_search_tool],
  memory=memory,
  verbose=True,
  handle_parsing_errors=True
)
# Tìm giá vé bằng google search hơi cùi
while True:
    memory.clear()
    question = input("Nhập câu hỏi: ")
    if question.lower() in ["exit", "quit"]:
        break
    response = agent_price_executor.invoke({"input": question})
    final_answer = response["output"]
    print(final_answer)
