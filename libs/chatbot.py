from langchain_groq import ChatGroq
import logging
from libs.prompt_template import *
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os


### Import thư viện
import logging
import os
import dotenv
import re
import json
from typing import List
from json import dumps, loads
from libs.prompt_template import *
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

### Cấu hình
dotenv.load_dotenv("api.env")
api_key_llm = os.getenv("GROQ_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Question Processing")
current_dir = os.path.dirname(os.path.abspath(__file__))


# Đối tượng chatbot
class Chatbot:
    def __init__(self):
        self.llm = ChatGroq(
            model_name="deepseek-r1-distill-qwen-32b",
            temperature=0.7,
            api_key=api_key_llm
        )
        ### Biến tóm tắt cuộc trò chuyện
        self.history_conversation = ""
        ### Biến truy vấn ban đầu
        self.query = ""
        ### embedding_model
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        ### vectorstore
        self.vectorstore = Chroma(
            persist_directory= os.path.join(current_dir, "chroma_db"),
            embedding_function= self.embedding_model,
        )
        ### Tạo retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )
        ### Tạo memory cho các hàm gọi tool có thể dùng
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        ### Tạo chain tạo nhiều câu hỏi
        self.query_generation_chain = LLMChain(
            llm=self.llm,
            prompt=query_generation_prompt_template
        )

        # Định nghĩa tool TravelAgent
        self.tool_memory_agent = Tool(
            name="MemoryAgent",
            func=self.query_history,
            description="Tìm kiếm thông tin mà người dùng cung cấp hoặc trong cuộc trao đổi trong lịch sử trò chuyện."
        )

        # Định nghĩa tool HotelSearchAgent
        self.tool_travel_agent = Tool(
            name="TravelAgent",
            func=self.refine_query,
            description="Tìm kiếm khách sạn dựa trên địa điểm, ngày nhận phòng và ngày trả phòng."
        )

        # Khởi tạo React Agent
        self.react_agent = create_react_agent(
            llm=self.llm,
            tools=[self.tool_memory_agent, self.tool_travel_agent],
            prompt=main_prompt_template
        )

        # Tạo AgentExecutor
        self.agent_search_executor = AgentExecutor(
            agent=self.react_agent,
            tools=[self.tool_memory_agent, self.tool_travel_agent],
            memory= self.memory,
            verbose=True,
            handle_parsing_errors=False
        )

    
    def get_query(self, query):
        self.query = query
    

    ### Hàm refine query dựa trên lịch sử hội thoại
    def refine_query(self, user_query):
        """
        Refine câu hỏi của người dùng dựa trên lịch sử hội thoại, sử dụng prompt mới với phương pháp Chain of Thought (CoT).
        """
        print(f"TravelAgent được gọi với query: {user_query}")
        
        # Hàm định dạng prompt với đầu vào từ người dùng
        def format_prompt(inputs):
            return query_refinement_prompt_template.format(**inputs)

        # Chuỗi xử lý refine query
        refined_query_chain = (
            RunnableLambda(format_prompt)  # Chuyển PromptTemplate thành Runnable
            | self.llm  # Gọi Large Language Model
            | StrOutputParser()  # Chuyển output về dạng string
        )
        
        print(self.history_conversation)
        refined_query = refined_query_chain.invoke({
            "chat_history": self.history_conversation, 
            "user_query": user_query
        })

        # Kiểm tra nếu refined_query là dictionary, trích xuất phần text
        if isinstance(refined_query, dict):
            refined_query = refined_query.get("text", "")
        
        # Nếu kết quả chứa "<ASK>", trả về nguyên đoạn đó
        if "<ASK>" in refined_query:
            return refined_query

        # Nếu không có "<ASK>", tiếp tục xử lý các bước tiếp theo
        return self.handle_next_steps(refined_query)

    ###Tóm tắt lịch sử
    def summary_history(self, conversation):
        summary_history_chain = (
            summary_history_prompt_template
            | self.llm  # Sử dụng Large Language Model để sinh output
            | StrOutputParser()  # Parser để chuyển đổi output về dạng string
        )
        response = summary_history_chain.invoke({"history": self.history_conversation, "new_conversation": conversation})
        return response


    ###Truy vấn trong lịch sử
    def query_history(self, query):
        print(f"MemoryAgent được gọi với query: {query}")
        query_history_chain =(
            query_history_prompt_template
            | self.llm
            | StrOutputParser() 
        )
        response = query_history_chain.invoke({"history": self.history_conversation, "query": query})
        return response

    def remove_think(self, response):
        """Loại bỏ phần <think> và chỉ giữ lại JSON"""
        if not isinstance(response, str):
            print(f"Lỗi: response không phải là chuỗi ở think! Dữ liệu nhận được: {type(response)}")
            return ""
        
        cleaned_response = re.sub(r'<think>[\s\S]*?</think>', '', response)
        cleaned_response = re.sub(r'```json|```', '', cleaned_response).strip()
        cleaned_response = cleaned_response.replace("\n", "").strip()
        return cleaned_response
    
    def remove_only_think(response):
        """Loại bỏ nội dung bên trong <think>...</think> mà vẫn giữ nguyên dấu xuống dòng."""
        return re.sub(r'<think>[\s\S]*?</think>', '', response, flags=re.DOTALL)
    
    def query_generation(self, user_input):
        response = self.query_generation_chain.invoke({"question": user_input})
        response = response.get("text", "")
        response = self.remove_think(response)
        response_json = json.loads(response)  # Không có try-except, lỗi JSON sẽ dừng chương trình
        variants = response_json.get("variant_questions", [])
        
        print("Các câu hỏi biến thể:")
        for q in variants:
            print(q)
        
        return variants
    
    def reciprocal_rank_fusion(self, results: list[list], k=100, top_n=4):
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_dict = doc.to_dict() if hasattr(doc, "to_dict") else doc.__dict__
                doc_str = dumps(doc_dict)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc_str), score)
            for doc_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results[:top_n] if top_n else reranked_results
    
    def generate_response(self, user_input, prompt_template_for_query):
        cleaned_query_generation = RunnableLambda(self.query_generation)
        
        retrieval_chain_rag_fusion = (
            cleaned_query_generation
            | self.retriever.map()
            | RunnableLambda(self.reciprocal_rank_fusion)
        )
        
        def format_input(inputs):
            context_result = retrieval_chain_rag_fusion.invoke(inputs)
            return {"retrieved_context": context_result, "question": inputs["question"]}
        
        formatted_prompt = RunnableLambda(lambda x: prompt_template_for_query.format(**x))
        
        final_rag_chain = (
            RunnableLambda(format_input)
            | formatted_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return final_rag_chain.invoke({"question": user_input})

    
    ### Hàm lấy thông tin tìm kiếm
    def handle_next_steps(self, query):
        """
        Thực hiện các bước tiếp theo sau khi refine câu hỏi.
        """
        response = self.generate_response(query, location_info_prompt)
        return response  # Thay thế bằng xử lý thực tế của bạn
    
    def chat(self, user_input):
        self.get_query(user_input)
        response = self.refine_query(user_input)
        return response
