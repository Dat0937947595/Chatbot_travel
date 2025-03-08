import os
import logging

from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv
from config.config import DATA_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL, LST_LLMS

# ============== Query Transformation ============== #

# Load environment variables
load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("====== Chatbot Travel ======")

# ====================================== #
# ============== Cấu hình ============== #

# Khởi tạo mô hình embedding
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
logger.info("Mô hình embedding được khởi tạo thành công.")
