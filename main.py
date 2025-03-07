from libs.chatbot import Chatbot

chatbot = Chatbot()

while True:
    user_input = input("Bạn: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chatbot.chat(user_input)
    print("Chatbot:", response)
