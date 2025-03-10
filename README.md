# CHATBOT_TRAVEL

## Cấu trúc dự án

### /config
- **Mục đích**: Lưu trữ các tệp cấu hình cho dự án.  
- **`config.py`**: Chứa các biến cấu hình toàn cục, ví dụ: `VECTORSTORE_DIR` để chỉ định thư mục lưu vector.

### /data
- **Mục đích**: Lưu trữ dữ liệu thô và đã xử lý dùng cho chatbot (dữ liệu du lịch, log truy vấn).

### /imgs
- **Mục đích**: Chứa các hình ảnh tĩnh (icon, giao diện) của dự án.

### /notebooks
- **Mục đích**: Lưu trữ các file Jupyter Notebook để thử nghiệm hoặc ghi lại quá trình phát triển.

### /prompts: 
- **Mục đích**:Quản lý các mẫu prompt (ví dụ: prompt tinh chỉnh truy vấn, prompt trả lời thông tin địa điểm).  

### /src
- **Mục đích**: Thư mục chính chứa mã nguồn của chatbot.  
- **`chatbot.py`**: Định nghĩa lớp `Chatbot`, khởi tạo mô hình ngôn ngữ, vector store, và xử lý tương tác người dùng.  

- **`services.py`**: Chứa logic xử lý nâng cao như tinh chỉnh truy vấn, truy vấn lịch sử, và tạo phản hồi.  
- **`utils.py`**: Chứa các hàm tiện ích như `remove_think` (xóa thẻ `<think>`), `reciprocal_rank_fusion` (hợp nhất kết quả tìm kiếm).

### /test_code
- **Mục đích**: Test các chức năng của hàm, test code

### /vectorstores
- **Mục đích**: Lưu trữ dữ liệu vector hóa (ví dụ: Chroma database) dùng cho tìm kiếm.

### File ở thư mục gốc
- **`main.py`**: Điểm chạy chính của ứng dụng, khởi tạo `Chatbot` và xử lý đầu vào người dùng.  
- **`requirements.txt`**: Danh sách các thư viện cần thiết để chạy dự án.