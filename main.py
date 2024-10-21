import os
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Define the directory containing the text file and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data")
persistent_dir = os.path.join(current_dir, 'vectorstores', 'db_chroma')


# Load the Ollama embeddings
embeddings = OllamaEmbeddings(
    model='nomic-embed-text'
)

# Load Chroma vector store
vector_db = Chroma(
    persist_directory=persistent_dir,
    embedding_function=embeddings
)

# Define the query
query = "Phân loại học bổng."

# Retrieve the relevant documents based on the query
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


# # Khởi tạo LLM với mô hình Llama2
# llm = ChatOllama(model='qwen2.5:7b', temperature=0.7)

# # Ví dụ truy vấn
# query = "Hãy sắp xếp cho tôi một lịch trình 2 ngày đi du lịch tại Đà Nẵng với 2 điểm du lịch là Chùa Linh Ứng và Bà Nà Hills."
# response = llm.invoke(query)

# # Hiển thị kết quả
# print("Response:", response)
