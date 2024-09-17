from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from scr.base.backend.rag.vector_search import get_search_result
from langchain.chains import RetrievalQA
from scr.base.backend.rag.store_FAISS import create_db_from_text
from langchain_community.vectorstores import FAISS
from path import PATH

path = PATH()
def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type = "llama",
        max_new_token = 1024,
        temperature = 0.1,
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables = ["context","question"])
    return prompt

def create_simple_chain(prompt, llm):
    llm_chain = LLMChain(prompt = prompt , llm = llm)
    return llm_chain

template = """<|im_start|>system\nBạn là một trợ lí AI du lịch hữu ích. Sử dụng thông tin sau đây để trả lời câu hỏi nếu cần thiết. Nếu thấy không cần thiết thì có thể bỏ qua. Nếu không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.\n{context}\n<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
# template = """<|im_start|>system
# Bạn là một trợ lí AI du lịch hữu ích. Hãy trả lời người dùng một cách chính xác những câu hỏi liên quan đến du lịch. Nếu gặp câu hỏi khác hãy giải thích bạn không trả lời những câu hỏi khác.
# <|im_end|>
# <|im_start|>user
# {question}<|im_end|>
# <|im_start|>assistant
# """

#khong co RAG
# def answer_str(query):
#     prompt = create_prompt(template)
#     llm = load_llm(model_file)

#     llm_chain = create_simple_chain(prompt, llm)
#     print(type(query))
#     #information = get_search_result(query, collection_input="tourist_location")
#     #response = llm_chain.invoke({"question":query, "context": information})
#     response = llm_chain.invoke({"question":query})
#     print(response)
#     answer = response.get('text', '')
#     return answer

#co RAG
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}),
        return_source_documents = False,
        chain_type_kwargs = {"prompt": prompt}
    )
    return llm_chain

def read_db_FAISS():
    embedding_model = path.embedding_model()
    db = FAISS.load_local(path.Faiss, embedding_model, allow_dangerous_deserialization=True)
    return db

def answer_str(query):
    print("1")
    create_db_from_text(get_search_result(query, collection_input= "tourist_location"))
    print("2")
    llm = load_llm(path.llm)
    db = read_db_FAISS()
    print("3")
    prompt = create_prompt(template)
    print("4")
    llm_chain = create_qa_chain(prompt, llm, db)
    print("5")
    response = llm_chain.invoke({"query": query})
    print(response)
    answer = response.get('result', '')
    return answer


# query = "Đặc điểm của Cà Mau"
# collection = "tourist_location"
# source_infomation = get_search_result(query, collection)
# print(source_infomation)
# read_text = create_db_from_text(source_infomation)
# print("1")
# llm = load_llm(path.llm)
# db = read_db_FAISS()
# print("3")
# prompt = create_prompt(template)
# print("4")
# llm_chain = create_qa_chain(prompt, llm, db)
# print("5")
# response = llm_chain.invoke({"query": query})
# # Lấy phần 'result' từ response
# result_text = response.get('result', '')

# # In ra màn hình với ký tự xuống dòng
# print(result_text)

