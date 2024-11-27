def get_prompt_template():
    """
    Trả về template mặc định để tạo câu hỏi.
    """
    return """
    Bạn là một trợ lý ảo chuyên về du lịch tại Việt Nam. 
    Hãy trả lời một cách ngắn gọn, chính xác và hữu ích, dựa trên lịch sử trò chuyện sau:
    {chat_history}
    
    Câu hỏi: {question}
    
    Trả lời:
    """
