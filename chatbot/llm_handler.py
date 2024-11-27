from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

class LLMHandler:
    """
    Lớp xử lý logic giao tiếp với mô hình Qwen 2.5-7B thông qua ChatOllama
    và sử dụng ChatPromptTemplate để quản lý prompt.
    """
    def __init__(self, model_name="qwen2.5:7b", temperature=0.7):
        """
        Khởi tạo LLMHandler với mô hình Ollama.
        
        Args:
            model_name (str): Tên của mô hình.
            temperature (float): Độ sáng tạo trong câu trả lời.
            max_tokens (int): Giới hạn số lượng token cho đầu ra.
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Bạn là trợ lý du lịch thông minh. Hãy trả lời câu hỏi một cách chính xác, đầy đủ thông tin, hữu ích và ngắn gọn."),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])

    def generate_answer(self, question, chat_history):
        """
        Sinh câu trả lời từ mô hình.

        Args:
            question (str): Câu hỏi từ người dùng.
            chat_history (list): Lịch sử hội thoại dưới dạng danh sách message.

        Returns:
            str: Câu trả lời từ mô hình.
        """
        # Tạo prompt từ template
        prompt = self.prompt_template.format_messages(
            chat_history=chat_history,
            question=question
        )
        
        # Gửi prompt đến mô hình và nhận phản hồi
        response = self.llm.invoke(prompt)
        return response.content
