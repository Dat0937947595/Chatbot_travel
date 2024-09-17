from sentence_transformers import SentenceTransformer

model_embedd_path = "./all-MiniLM-L6-v2-f16.gguf"
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embedding_vector(text: str) -> list[float]:
    embed = embedding_model.encode(text).tolist()
    return embed

    