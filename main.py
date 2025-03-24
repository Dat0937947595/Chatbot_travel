from src.chatbot import Chatbot

# Chạy thử chatbot
if __name__ == "__main__":
    chatbot = Chatbot(verbose=True)  # Bật verbose để debug
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chatbot.chat(user_input)
        print(f"Chatbot: {response}")
        # Reset bộ nhớ nếu cần
    chatbot.reset_memory()
    