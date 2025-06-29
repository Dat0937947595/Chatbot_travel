import streamlit as st
import logging
import base64
from src.chatbot import Chatbot
import time
import threading

# Cấu hình log
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    if st.session_state.get("resetting", False):
        # Xóa cờ và rerun trước khi vẽ giao diện
        del st.session_state["resetting"]
        st.rerun()


    # Cấu hình trang
    st.set_page_config(
        page_title="Chatbot Q&A Assistant",
        page_icon="imgs/bot.png",
        layout="wide",
        initial_sidebar_state="auto"
    )

    # Hiển thị tiêu đề
    st.title("Chatbot Assistant")

    # Hàm chuyển ảnh sang base64
    def img_to_base64(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception as e:
            logging.error(f"Lỗi chuyển ảnh: {str(e)}")
            return None

    # Khởi tạo chatbot và session state
    @st.cache_resource
    def init_chatbot():
        return Chatbot(verbose=False)

    chatbot = init_chatbot()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = True
        st.session_state.history.append({
            "role": "assistant",
            "content": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"
        })
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    # CSS làm đẹp hình ảnh avatar và vòng tròn loading
    st.markdown("""
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #665500,
                0 0 10px #997700,
                0 0 15px #CCAA00,
                0 0 20px #FFCC00,
                0 0 25px #FFDD33,
                0 0 30px #FFEE66,
                0 0 35px #FFFF99;
            position: relative;
            z-index: -1;
            border-radius: 45px;
            margin-bottom: 40px;
        }
        .chat-element {
            margin-bottom: 24px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #FF6666;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 10px 0;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Hiển thị sidebar ---
    img_path = "imgs/bot.png"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">', 
            unsafe_allow_html=True
        )

    st.sidebar.markdown("\n---\n")
    st.sidebar.markdown("## Chào mừng bạn đến với chatbot hỏi đáp về du lịch !\n")
    st.sidebar.markdown("### Hướng đẫn:")
    st.sidebar.markdown("Nhập câu hỏi của bạn vào hộp thoại bên cạnh để bắt đầu.")
    st.sidebar.markdown("---")

    # Hiển thị lịch sử chat
    def display_chat_history():
        for msg in st.session_state.history:
            role = msg["role"]
            content = msg["content"]
            avatar = (
                "imgs/bot.png" if role == "assistant"
                else "imgs/stuser.png"
            )
            if role == "user":
                st.markdown(f"""
                    <div style='
                        display: flex;
                        justify-content: flex-end;
                        margin-bottom: 3px;
                    '>
                        <div style='
                            background-color: #DCF8C6;
                            padding: 12px 16px;
                            border-radius: 16px;
                            max-width: 80%;
                            text-align: left;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        '>
                            {content}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            elif role == "assistant":
                with st.chat_message("assistant", avatar=avatar):
                    st.markdown(f"""
                        <div style='
                            background-color: #f0f2f6;
                            padding: 12px 16px;
                            border-radius: 12px;
                            max-width: 90%;
                            margin-bottom: 16px;
                            display: inline-block;
                        '>{content}</div>
                    """, unsafe_allow_html=True)
    
    # Hiển thị đoạn tin nhắn AI cuối cùng
    def display_last_ai_message():
        # Hiển thị assistant message cuối cùng với hiệu ứng gõ chữ
        if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
            msg = st.session_state.history[-1]
            content = msg["content"]

            with st.chat_message("assistant", avatar="imgs/bot.png"):
                typewriter_area = st.empty()
                typed_text = ""
                for char in content:
                    typed_text += char
                    typewriter_area.markdown(f"""
                        <div style='
                            background-color: #f0f2f6;
                            padding: 12px 16px;
                            border-radius: 12px;
                            max-width: 90%;
                            margin-bottom: 16px;
                            display: inline-block;
                            font-size: 16px;
                        '>{typed_text}</div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.015)
            
        
    # Hộp chat input
    user_input = st.chat_input("Nhập câu hỏi của bạn ở đây ...", disabled=st.session_state.is_processing)
    if user_input:
        st.session_state.is_processing = True
        st.session_state.history.append({"role": "user", "content": user_input})
        display_chat_history()

        loading_area = st.empty()
        result = [None]
        start_time = time.time()

        # Thread để xử lý chatbot
        def ask_bot():
            result[0] = chatbot.chat(user_input)

        thread_chat = threading.Thread(target=ask_bot)
        thread_chat.start()

        # Cập nhật loading và thời gian trong main thread
        while thread_chat.is_alive():
            elapsed = time.time() - start_time
            loading_area.markdown(f"""
                <div style='text-align:left; margin-top: 16px;'>
                    <div class="spinner"></div>
                    <p class="thinking-text" style="font-size:16px;">Đang suy nghĩ... ({elapsed:.1f}s)</p>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(0.1)

        st.session_state.is_processing = False
        loading_area.empty()

        st.session_state.history.append({"role": "assistant", "content": result[0]})
        display_last_ai_message()
    else:
        # Hiển thị lịch sử chat khi không có input mới
        display_last_ai_message()

    # Nút reset
    if st.sidebar.button("🔁 Reset hội thoại"):
        chatbot.reset_memory()
        st.session_state.clear()
        st.session_state["resetting"] = True
        st.stop()  # Dừng mọi render tiếp theo → tránh hiện thông tin cũ