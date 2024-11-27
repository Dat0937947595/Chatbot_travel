import streamlit as st
from chatbot.llm_handler import LLMHandler
from config.settings import APP_TITLE

# Thiết lập giao diện Streamlit
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"💬 {APP_TITLE}")

# Tạo LLM handler
llm_handler = LLMHandler()

# Khởi tạo session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "waiting_for_response" not in st.session_state:
    st.session_state["waiting_for_response"] = False

# Hiển thị hội thoại
for msg in st.session_state["chat_history"]:
    role = "👤" if msg["role"] == "user" else "🤖"
    st.chat_message(role).write(msg["content"])

# Nhập câu hỏi
if not st.session_state.waiting_for_response:
    prompt = st.chat_input(placeholder="Nhập câu hỏi của bạn tại đây...")
    if prompt:
        # Thêm câu hỏi vào lịch sử
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        st.chat_message("👤").write(prompt)

        # Xử lý trả lời
        st.session_state.waiting_for_response = True
        with st.spinner("Đang xử lý câu trả lời..."):
            answer = llm_handler.generate_answer(
                question=prompt,
                chat_history=st.session_state["chat_history"]
            )
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            st.chat_message("🤖").write(answer)

        st.session_state.waiting_for_response = False
