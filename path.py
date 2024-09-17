from start_mongoDB import start_mongo
from langchain_community.embeddings import GPT4AllEmbeddings

class PATH:
    def __init__(self) -> None:
        self.embedd = "scr/model_ai/all-MiniLM-L6-v2-f16.gguf"
        self.llm = "scr/model_ai/vinallama-7b-chat_q5_0.gguf"
        self.Faiss = "scr/base/backend/vectorstore/faiss_db"
        self.mongDB = start_mongo()
    
    def embedding_model(self):
        embedding_model = GPT4AllEmbeddings(model_file = self.embedd)
        return embedding_model

        
    
