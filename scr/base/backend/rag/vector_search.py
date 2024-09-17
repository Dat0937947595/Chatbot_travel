
# from scr.base.backend.model_ai.embed import embedding_vector
from path import PATH

path = PATH()

def embedding_vector(text: str) -> list[float]:
    embedding_model = path.embedding_model()
    embed = embedding_model.embed_query(text)
    return embed
# Kết nối tới MongoDB

def vector_search(user_query, collection_input):
    collection = path.mongDB[collection_input]
    #embedding query
    query_embedding = embedding_vector(user_query)
    
    if query_embedding is None:
        return "Invalid query or embedding generation failed."
    
    #Define the vector search pipeline
    vector_search_stage = {
        "$vectorSearch":{
            "index": "default",
            "queryVector": query_embedding,
            "path": "vector",
            "numCandidates": 150, #gioi han khong gian
            "limit": 4 #tra ve 4 cai match nhat 
        }
    }

    unset_stage = {
        "$unset": "vector"
    }

    project_stage = {
        "$project": {
            "_id": 0,
            "text": 1,
            "score": {
                "$meta": "vectorSearchScore" # them diem so tuong dong de quan sat
            }
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]

    #execute the search
    results = collection.aggregate(pipeline)
    return list(results)

def get_search_result(query, collection_input):
    get_knowledge = vector_search(query, collection_input)

    search_result = ""
    seen_texts = set()  # Tập hợp để theo dõi các văn bản đã thấy

    for result in get_knowledge:
        text = result.get('text')
        if text not in seen_texts:
            search_result += f"{text}\n"
            seen_texts.add(text)  # Thêm văn bản vào tập hợp đã thấy

    return search_result

#use
#truyen cai nay di