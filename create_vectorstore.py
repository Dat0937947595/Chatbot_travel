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

from config import DATA_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL
from prompt import qa_prompt_system, query_generation_5_prompt_template, contextualize_q_prompt

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Create Vectorstore ...")

# ============== Functions =============== #
# Hàm load file .txt và .md
def load_documents(data_path):
    # Tải tài liệu từ thư mục dữ liệu. 
    documents = []
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_name.endswith(".md"):
            loader = TextLoader(file_path, encoding="utf-8") # Ban đầu dùng UnstructuredMarkdownLoader nhưng nó bị mất ksi hiệu bảng
        else:
            continue
        documents.extend(loader.load())
    return documents

# Chunking dữ liệu bằng SemanticChunker
def chunk_documents(documents, embedding_model):
    # Sử dụng Semantic Chunker để chia nhỏ tài liệu dựa trên ngữ nghĩa.
    semantic_chunker = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")
    chunks = []
    for doc in documents:
        split_texts = semantic_chunker.create_documents([doc.page_content])
        for chunk in split_texts:
            chunks.append(Document(page_content=chunk.page_content, metadata=doc.metadata))
        
    return chunks

# Tạo vectorstore
def create_vectorstore(documents, vectorstore_dir, embedding_model):
    # Tạo vector store để lưu trữ các đoạn văn bản đã embedding.
    if not os.path.exists(vectorstore_dir):
        os.makedirs(vectorstore_dir)
    logger.info("Đang tạo vector store...")
    vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=vectorstore_dir)
    return vectorstore

# ============== Vectorstore ============== #

# Khởi tạo mô hình embedding
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
logger.info("Mô hình embedding được khởi tạo thành công.")

# Tạo vectorstore nếu chưa tồn tại
if not os.path.exists(VECTORSTORE_DIR):
    logger.info("Vector store không tồn tại. Đang tạo mới...")
    
    # Load documents
    logger.info("Đang tải tài liệu...")
    documents = load_documents(DATA_DIR)
    logger.info(f"Số lượng tài liệu: {len(documents)}")
    
    # Chunk documents
    logger.info("Đang chia nhỏ tài liệu...")
    chunked_documents = chunk_documents(documents, embedding_model)
    logger.info(f"Số lượng tài liệu sau khi chia nhỏ: {len(chunked_documents)}")
    
    # Create vectorstore
    logger.info("Đang tạo vector store...")
    vectorstore = create_vectorstore(chunked_documents, VECTORSTORE_DIR, embedding_model)
    logger.info("Vector store đã được tạo và lưu thành công.")
else:
    logger.info("Vector store đã tồn tại.")