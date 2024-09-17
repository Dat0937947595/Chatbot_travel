# vector_store.py
from pymongo import MongoClient
from langchain_community.embeddings import GPT4AllEmbeddings
import os

# Kết nối tới MongoDB
client = MongoClient('mongodb+srv://minhdat:1111@travelchatbot.izjgp.mongodb.net/')
db = client['travel_db']
collection = db['tourist_location']
model_embedd_path = "scr/model_ai/all-MiniLM-L6-v2-f16.gguf"

# Load a pre-trained sentence transformer model local
#embedding_model = GPT4AllEmbeddings(model_file = model_embedd_path)

#Load từ hugging face
embedding_model = GPT4AllEmbeddings(model_file = model_embedd_path)

# # Đọc file PDF và trích xuất văn bản
# pdf_path = 'path_to_your_file.pdf'  # Cập nhật đường dẫn tới file PDF của bạn
# documents = []


# with pdfplumber.open(pdf_path) as pdf:
#     for page in pdf.pages:
#         text = page.extract_text()
#         if text:  # Kiểm tra nếu trang không rỗng
#             documents.append(text)

# # Insert vectors into MongoDB
# for text in documents:
#     vector = model.encode(text).tolist()
#     collection.insert_one({"text": text, "vector": vector})

# # Đọc file .txt và trích xuất văn bản
# txt_path = 'scr/base/backend/vectorstore/HCM_du_lich.txt'  # Cập nhật đường dẫn tới file .txt của bạn

# with open(txt_path, 'r', encoding='utf-8') as file:
#     text = file.read()

# # Tạo và lưu trữ embedding
# vector = embedding_model.encode(text).tolist()
# collection.insert_one({"text": text, "vector": vector})
# Thư mục chứa các file .txt
txt_directory = 'scr/base/backend/vectorstore/data_source/data_generative'  # Cập nhật đường dẫn tới thư mục chứa các file .txt

# Duyệt qua toàn bộ các file trong thư mục
for filename in os.listdir(txt_directory):
    if filename.endswith(".txt"):  # Kiểm tra nếu file là .txt
        txt_path = os.path.join(txt_directory, filename)  # Tạo đường dẫn đầy đủ tới file .txt

        # Đọc file .txt và trích xuất văn bản
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Tạo và lưu trữ embedding
        vector = embedding_model.embed_query(text).tolist()
        collection.insert_one({"filename": filename, "text": text, "vector": vector})

        print(f"Đã lưu trữ embedding cho file: {filename}")