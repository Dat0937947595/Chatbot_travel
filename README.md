# Chatbot Travel Assistant

## Cách chạy ứng dụng

### Chạy trực tiếp bằng Streamlit

```bash
streamlit run app.py
```

Ứng dụng sẽ được khởi chạy tại địa chỉ:

- Local URL: http://localhost:8501

- Network URL: http://192.168.2.4:8501

### Chạy bằng Docker

1. **Build image:**
```bash
docker build -t chatbot_app .
```

2. **Run container:**
```bash
docker run -p 8501:8501 chatbot_app
```

Sau khi chạy thành công, truy cập tại: [http://localhost:8501](http://localhost:8501)

> **Lưu ý:** Nếu bạn dùng Docker Desktop, đảm bảo cổng 8501 chưa bị chiếm và Docker đã khởi động.

---

## Cấu trúc dự án

```
CHATBOT_TRAVEL/
├── config/                  # Cấu hình cho ứng dụng
├── imgs/                    # Chứa hình ảnh minh họa, avatar cho UI Streamlit
├── prompts/                 # Lưu các prompt mẫu sử dụng cho mô hình LLM
├── src/                     # Code logic chính của hệ thống chatbot
│   ├── __init__.py          # Đánh dấu thư mục là Python package
│   ├── chatbot.py           # Class hoặc hàm xử lý hội thoại chính của chatbot
│   ├── model.py             # Khởi tạo với mô hình (llm, embeddings)
│   ├── services.py          # Các dịch vụ xử lý trung gian (ví dụ xử lý câu trả lời, gọi API phụ)
│   └── utils.py             # Hàm tiện ích, xử lý dữ liệu hỗ trợ các module khác
├── test_code/               # Chứa các file test cho logic của chatbot (unit test, integration test)
├── vectorstores/            # Dùng cho lưu trữ vector trong các hệ thống RAG hoặc semantic search

├── .dockerignore            # File chỉ định các file/folder cần bỏ qua khi build Docker image
├── .env                     # Biến môi trường (API key, cấu hình bảo mật,...)
├── .gitignore               # Bỏ qua các file không cần thiết khi push lên Git (như __pycache__, .env, v.v.)

├── Dockerfile               # File cấu hình để build Docker image cho ứng dụng
├── main.py                  # Entry point khác nếu không dùng Streamlit (ví dụ chạy bằng CLI hoặc FastAPI)
├── README.md                # File hướng dẫn tổng quan dự án, cách sử dụng, cấu hình,...
└── streamlit_app.py         # File chính khởi chạy ứng dụng Streamlit (frontend chatbot)
```