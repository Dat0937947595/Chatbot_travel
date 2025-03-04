# Về Dataset
* Dataset bao gồm Chương trình đào tạo Chính Quy Khóa tuyển 2023 của các ngành Công nghệ thông tin, Khoa học máy tính, Hệ thống thông tin, Kỹ thuật phần mềm, Trí tuệ nhân tạo và Cử nhân tài năng. 
* Ngoài ra, dataset còn chứa thông tin về Sổ tay sinh viên bao gồm các quy chế học vụ, điểm rèn luyện,...

## Về phương pháp thu thập dữ liệu

* Dữ liệu về chương trình đạo tạo được khoa đăng công khai tại [liên kết này](https://www.fit.hcmus.edu.vn/vn/Default.aspx?tabid=1284) và dữ liệu Sổ tay sinh viên được trường đăng [tại đây](https://hcmus.edu.vn/so-tay-sinh-vien-online-nam-hoc-2023-2024/)
* Tuy nhiên, các file chương trình đào tạo là file scan được xuất ở định dạng .pdf. Nhóm đã sử dụng công cụ [LlamaCloud](https://cloud.llamaindex.ai/) để xài tính năng [LlamaParse](https://cloud.llamaindex.ai/parse) nhằm chuyển đổi file .pdf thành file text với định dạng markdown (.md). (Tính năng này đã được thiết kế để có hiệu suất tốt với RAG).
* File Sổ tay sinh viên, vừa sử dụng công cụ [LlamaParse](https://cloud.llamaindex.ai/parse) sau đó chỉnh sửa thủ công.
* Bộ Test được xây dựng thủ công một phần, phần còn lại prompt cho LLMs (ở đây là ChatGPT) để sinh ra các câu hỏi và câu trả lời; đồng thời nhóm sẽ đánh giá câu hỏi và câu trả lời nào hợp lý để quyết định cho vào bộ Test hay loại bỏ.

### Tổ chức project

```plaintext
EduBot/
├── README.md           # Tài liệu hướng dẫn chạy chương trình
├── app.py              # Tệp chính chạy ứng dụng Streamlit
├── requirements.txt    # Danh sách thư viện cần thiết
├── configs/            # Cấu hình ứng dụng
│   └── settings.py     # File cấu hình chính
├── data/               # Thư mục chứa dữ liệu
│   ├── docs/           # Các tài liệu sử dụng 
│   ├── test/           # Bộ dữ liệu kiểm tra mô hình
│   └── readme.md       # Mô tả về dataset
├── src/                # Chứa các mã nguồn sử dụng
│   ├── base/           # Các mã nguồn dùng để khởi tạo mô hình
│   └── rag/            # Các mã nguồn cài đặt và vận hành RAG
│       └── chatbot.py  # Mã nguồn kiến trúc của chatbot
└── vectorstores/       # Chứa các vector embeddings từ văn bản
```

### Định dạng các file dữ liệu

```plaintext
- .txt  : FAQ và Sổ tay sinh viên
- .md   : Chương trình đào tạo
- .xlsx : Bộ test
```

## Liên kết đến Dataset online

* [Data Repository](https://github.com/ncc2k3/EduBot/tree/main/data)

## Authors

* *21120149* -  **Nguyễn Đăng Thới Toàn**
* *21120422* -  **Nguyễn Chí Cường**
* *21120602* - **Võ Ngọc Trí** 


## Giấy phép

- MIT License
- Bản quyền (c) **[2024] Group01 - Chatbot RAG Project**
- Dữ liệu được thu thập từ chương trình đào tạo khoa Công nghệ thông tin và sổ tay sinh viên công khai của Trường Đại học Khoa học Tự nhiên - Đại học Quốc gia TP.HCM (HCMUS), được sử dụng với mục đích giáo dục và nghiên cứu. 


## Lời cảm ơn

- Khoa Công nghệ thông tin, Phòng công tác sinh viên của HCMUS.
- TS. Lê Thanh Tùng
- ThS. Nguyễn Trần Duy Minh