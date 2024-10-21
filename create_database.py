import os
from langchain.embeddings import HuggingfaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Đường dẫn tới tệp văn bản và vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data", "cleaned_STSV-2024.txt")
persistent_dir = os.path.join(current_dir, 'vectorstores', 'db_chroma')

# Khởi tạo Huggingface Embeddings với một model đa ngôn ngữ
embedding_model = HuggingfaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Hàm tùy chỉnh để đọc và chia nhỏ văn bản
def load_and_split_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400)
    chunks = splitter.split_text(content)

    return [Document(page_content=chunk, metadata={"source": file_path}) for chunk in chunks]

# Kiểm tra và tạo vector store nếu chưa có
if not os.path.exists(persistent_dir):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    print("\n--- Loading and splitting documents ---")
    docs = load_and_split_txt(file_path)

    # Tạo vector store và lưu lại
    print("\n--- Creating the Chroma vector store ---")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persistent_dir,
    )
    db.persist()
    print("\n--- Finished creating the Chroma vector store ---")

else:
    print(f"Persistent directory {persistent_dir} already exists.")
