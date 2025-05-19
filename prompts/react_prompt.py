from langchain_core.prompts import PromptTemplate
from datetime import datetime

prompt_react = PromptTemplate.from_template(
    """
    Trả lời các câu hỏi sau đây một cách tốt nhất có thể. Bạn có quyền truy cập vào các công cụ sau:
    {tools}
    Sử dụng định dạng sau:
    Question: câu hỏi đầu vào mà bạn phải trả lời
    Thought: bạn nên luôn suy nghĩ về việc cần làm gì
    Action: hành động cần thực hiện, nên là một trong số [{tool_names}]
    Action Input: đầu vào cho hành động
    Observation: kết quả của hành động
    ... (quy trình Thought/Action/Action Input/Observation này có thể lặp lại N lần)
    Thought: Bây giờ tôi đã biết câu trả lời cuối cùng
    Final Answer: câu trả lời cuối cùng cho câu hỏi đầu vào ban đầu. Câu trả lời thân thiện và tự nhiên. Format dễ nhìn dễ đọc.
    Begin!
    Thời gian hiện tại: {current_time}
    Question: {input}
    Thought:{agent_scratchpad}
    """
).partial(current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))