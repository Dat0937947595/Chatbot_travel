import os
import logging
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

import google.generativeai as genai
from dotenv import load_dotenv
from typing import List
from operator import itemgetter
import streamlit as st

from config.config import DATA_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL, LST_LLMS

# ============== Query Transformation ============== #
from prompt import qa_prompt_system, query_generation_5_prompt_template, contextualize_q_prompt

# Load environment variables
load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Edubot")

# ====================================== #
# ============== Cấu hình ============== #

# Khởi tạo mô hình embedding
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
logger.info("Mô hình embedding được khởi tạo thành công.")

# ======================================== #
# Load vectorstore
logger.info("Đang load vector store...")
vectorstore = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embedding_model
)

# Khởi tạo retriever
logger.info("Khởi tạo retriever...")
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}  
)

# ==================== RAG-Fusion ==================== #
from langchain.load import dumps, loads

### Reranking - Reciprocal Rank Fusion
def reciprocal_rank_fusion(results: list[list], k=100):
    # Reciprocal_rank_fusion that takes multiple lists of ranked documents 
    #    and an optional parameter k used in the RRF formula 
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results
    
### ============== Function ============== ###
class ListOutputParser(BaseOutputParser[List[str]]):
    # Output parser to extract only numbered questions. # 
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

output_parser = ListOutputParser()

### === Hàm phục vụ cho streamlit generate === ###
# Hàm refine query dựa trên lịch sử hội thoại
def refine_query(user_input, chat_history):
    """
    Refine câu hỏi của người dùng dựa trên lịch sử hội thoại, sử dụng prompt mới với phương pháp Chain of Thought (CoT).
    """
    try:
        logger.info(f"Refine query: {user_input}")
        
        # Thực hiện refine câu hỏi
        refined_query_chain = (
            contextualize_q_prompt
            | llm  # Sử dụng Large Language Model để sinh output
            | StrOutputParser()  # Parser để chuyển đổi output về dạng string
        )
        refined_query = refined_query_chain.invoke({"chat_history": chat_history, "input": user_input})
    except Exception as e:
        logger.error(f"Lỗi khi refine query: {e}")
        # Trường hợp lỗi, giữ nguyên câu hỏi gốc
        refined_query = user_input
    
    logger.info(f"Câu hỏi sau refine: {refined_query}")
    return refined_query


# Hàm sinh câu trả lời
def generate_response(user_input, llm):        
    query_generation_chain = query_generation_5_prompt_template | llm | output_parser

    # """ ============== Reranking  RAG-Fusion ============== """
    logger.info("Reranking RAG-Fusion...")
    retrieval_chain_rag_fusion = (
        query_generation_chain
        | retriever.map()  # Truy vấn từng câu hỏi trong danh sách
        | reciprocal_rank_fusion
    )
    
    logger.info("Sinh câu trả lời...")
    # Sử dụng LLM để sinh câu trả lời
    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion,
        "question": itemgetter("question"),} 
        | qa_prompt_system
        | llm
        | StrOutputParser()
    )
    return final_rag_chain.invoke({"question": user_input})
    

# Hàm main
if __name__ == "__main__":
    
    # Cấu hình giao diện Streamlit
    icon_url = "https://cdn-icons-png.flaticon.com/512/6540/6540769.png"
    st.set_page_config(page_title="Edubot", page_icon=icon_url)
    
    # Sidebar
    with st.sidebar:
        # Tạo tiêu đề với hình ảnh icon
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="{icon_url}" width="80" height="80" style="margin-right: 10px;">
                <h1 style="margin: 0 20; font-size: 2rem; line-height: 60px;">Edubot</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a LLMs model', LST_LLMS, key='selected_model')
            
        st.markdown("Tùy chỉnh tham số:")
        temperature = st.slider("Temperature", min_value=0.01, max_value=2.0, value=0.3, step=0.01)
        top_p = st.slider("Top P", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.slider("Max Length", min_value=64, max_value=1024, value=512, step=8)
        
        # Khởi tạo mô hình dựa trên lựa chọn
        if selected_model == 'llama3.1:latest':
            llm = ChatOllama(model="llama3.1:latest", temperature=temperature, top_p=top_p, max_length=max_length)
            logger.info("Khởi tạo mô hình llama3.1:latest thành công.")
        elif selected_model == 'gemini-1.5-flash':
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=temperature,
                top_p=top_p,
                max_length=max_length
            )
            logger.info("Khởi tạo mô hình Gemini-flash-1.5 thành công.")
        else:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=temperature,
                top_p=top_p,
                max_length=max_length
            )
            logger.info("Khởi tạo mô hình Gemini-pro-1.5 thành công.")
            
            
    # Khởi tạo lịch sử hội thoại
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi có thể giúp gì hôm nay?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi có thể giúp gì hôm nay?"}]
        logger.info("Đã xóa lịch sử hội thoại.")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Xử lý input từ người dùng
    if user_input := st.chat_input("Nhập câu hỏi của bạn"):
        # Sinh câu trả lời từ user_input sau khi refine query.
        # Refine câu hỏi dựa trên lịch sử
        logger.info("Refine query ...")
        refined_query = refine_query(user_input, st.session_state.messages)
        
        st.session_state.messages.append({"role": "user", "content": refined_query})
        with st.chat_message("user"):
            st.write(user_input)

        # Tạo câu trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                try:
                    response = generate_response(refined_query, llm)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    logger.info("Câu trả lời đã được sinh ra thành công.")
                except Exception as e:
                    st.error(f"Lỗi khi sinh câu trả lời: {e}")
                    logger.error(f"Lỗi khi sinh câu trả lời: {e}")