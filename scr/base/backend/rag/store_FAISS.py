from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from path import PATH

path = PATH()

def create_db_from_text(raw_text):
    #cat van ban
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len,
    )

    chunks = text_splitter.split_text(raw_text)

    #Embedding
    embedding_model = path.embedding_model()

    #Luu
    db = FAISS.from_texts(texts=chunks, embedding = embedding_model)
    db.save_local(path.Faiss)


#Để đó không dùng
# pđf_data_path = "data"
# def create_db_from_file():
#     #khai bao loader
#     loader = DirectoryLoader(pđf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(
#         separator = "\n",
#         chunk_size = 500,
#         chunk_overlap = 50,
#     )
#     chunks = text_splitter.split_documents(documents)

#     #Embedding
#     embedding_model = GPT4AllEmbeddings(model_file = model_embedd_path)

#     #Luu
#     db = FAISS.from_documents(chunks, embedding_model)
#     db.save_local(vector_db_path)
#     return db
