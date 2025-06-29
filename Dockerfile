# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /chatbot_app

# # Install dependencies
RUN pip install -U langchain langchain-chroma \
    langchain_core langchain_community langchain_google_genai \
    langchain_tavily 
RUN pip install tavily-python dotenv pydantic requests streamlit rank_bm25

COPY . .

# Run app
CMD ["streamlit", "run", "app.py"]