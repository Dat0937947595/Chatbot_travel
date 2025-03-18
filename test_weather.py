import requests
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key="AIzaSyA-MAlE62P8Gg2g664zwnYcRAtNykEg_tE"
        )

class WeatherAPIWrapper:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, location):
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "metric",  # Đơn vị độ C
            "lang": "vi"  # Ngôn ngữ tiếng Việt
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if response.status_code == 200:
                weather_desc = data["weather"][0]["description"].capitalize()
                temp = data["main"]["temp"]
                humidity = data["main"]["humidity"]
                wind_speed = data["wind"]["speed"]

                return (
                    f"Thời tiết tại {location}: {weather_desc}.\n"
                    f"Nhiệt độ: {temp}°C, Độ ẩm: {humidity}%, Gió: {wind_speed} m/s."
                )
            else:
                return f"Không tìm thấy thông tin thời tiết cho {location}."
        except Exception as e:
            return f"Lỗi khi gọi API thời tiết: {str(e)}"

# Thay API_KEY bằng khóa API thực của bạn
API_KEY = "b13f85eb589c453522bb1322a6763a8d"
weather_api = WeatherAPIWrapper(API_KEY)

from langchain.prompts import PromptTemplate

weather_prompt_template = PromptTemplate.from_template(
"""
Bạn là một trợ lý du lịch thông minh. Nhiệm vụ của bạn là cung cấp **thông tin thời tiết hiện tại** tại địa điểm do người dùng cung cấp.

Lịch sử hội thoại:
{chat_history}

Câu hỏi:
{input}

Công cụ có sẵn:
{tools}

Tên công cụ:
{tool_names}

Khi trả lời, hãy tuân theo các quy tắc sau:
1. **Chỉ cung cấp thông tin về thời tiết hiện tại.**
2. **Nếu người dùng hỏi về thời tiết trong quá khứ hoặc tương lai, hãy từ chối trả lời với thông báo lịch sự và hỏi lại người dùng có muốn tìm kiếm thông tin của địa điểm đó ở hiện tại không**
3. **Nếu bạn chưa có đủ thông tin để trả lời, hãy sử dụng công cụ để tìm kiếm thông tin thời tiết.**
4. **Không bao giờ trả về "Action" và "Final Answer" cùng lúc.**
5. **Không cần liệt kê các đường link tham khảo trong câu trả lời.**

### **Định dạng bắt buộc:**
Thought: Mô tả suy nghĩ của bạn về câu hỏi.
Action: <Tên công cụ> (chỉ khi cần tìm kiếm thêm thông tin)
Action Input: <Thông tin cần tìm kiếm chỉ bao gồm địa điểm>
Observation: <Thông tin thu được từ công cụ>

Final Answer: <Câu trả lời cuối cùng về thời tiết tại địa điểm yêu cầu. Bắt đầu bằng "Final Answer:">

---

### **Ví dụ:**

#### **Trường hợp hợp lệ (thời tiết hiện tại)**
**Câu hỏi:** "Thời tiết tại Hà Nội bây giờ thế nào?"

Thought: Tôi cần tìm thông tin thời tiết hiện tại tại Hà Nội.
Action: Weather Search
Action Input: Hà Nội
Observation: Hiện tại, Hà Nội có mưa nhẹ, nhiệt độ 25°C, độ ẩm 85%, gió tốc độ 10 km/h.

Final Answer: Thời tiết tại **Hà Nội ngay bây giờ**: **Mưa nhẹ**, nhiệt độ **25°C**, độ ẩm **85%**, gió **10 km/h**.

---

#### **Trường hợp bị từ chối (quá khứ hoặc tương lai)**
**Câu hỏi:** "Thời tiết tại Đà Nẵng vào ngày mai thế nào?"

Thought: Người dùng đang hỏi về dự báo thời tiết tương lai. Tôi chỉ có thể cung cấp thời tiết hiện tại, nên tôi cần từ chối.

Final Answer: Xin lỗi, tôi chỉ có thể cung cấp thông tin về **thời tiết hiện tại**. Tôi không thể tra cứu dữ liệu thời tiết trong tương lai. Bạn có muốn tôi tìm kiếm thông tin về thời tiết hiện tại ở Đà Nẵng không?

{agent_scratchpad}

STOP
"""
)

# Thiết lập tool
weather_tool = Tool(
    name="Weather Search",
    func=weather_api.get_weather,
    description="Tìm kiếm thông tin thời tiết hiện tại cho một thành phố hoặc địa điểm."
)
# Thêm vào danh sách công cụ
tools = [weather_tool]

# Khởi tạo ReAct Agent
react_agent = create_react_agent(
    llm=llm_gemini,  # LLM bạn đang dùng (ví dụ: OpenAI GPT, Llama, Groq)
    tools=tools,
    prompt=weather_prompt_template
)

# Tạo AgentExecutor để thực thi
agent_weather_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=False
)

memory.clear()
question = "Thời tiết ở Cần Thơ thế nào?"
response = agent_weather_executor.invoke({"input": question})
final_answer = response["output"]
print(final_answer)