from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from .utils import *
from prompts.prompt_template import *
from prompts.query_prompts import *
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Services") # Đổi tên logger từ "Query Processing" sang "Services"

### Hàm refine query dựa trên lịch sử hội thoại
def refine_query_processing(user_query, chat_history, llm):
	"""
	Refine câu hỏi của người dùng dựa trên lịch sử hội thoại, sử dụng prompt mới với phương pháp Chain of Thought (CoT).
	"""
	try:
		logger.info(f"Query: {user_query}")
		
		# Thực hiện refine câu hỏi
		refined_query_chain = (
			rewrite_query_prompt  # Sử dụng prompt để viết lại câu hỏi
			| llm  # Sử dụng Large Language Model để sinh output
			| StrOutputParser()  # Parser để chuyển đổi output về dạng string
		)
		
		refined_query = refined_query_chain.invoke({"chat_history": chat_history, "user_query": user_query})
	except Exception as e:
		logger.error(f"Lỗi khi refine query: {e}")
		# Trường hợp lỗi, giữ nguyên câu hỏi gốc
		refined_query = user_query
	
	return refined_query

def refine_query(chatbot, user_query):
    """Tinh chỉnh câu hỏi người dùng dựa trên lịch sử hội thoại."""
    refined_query = refine_query_processing(user_query, chatbot.history_conversation, chatbot.llm_gemini)
    logger.info(f"\nCâu hỏi sau refine: {refined_query}")

    # Kiểm tra trường hợp câu hỏi không rõ ràng - nếu có cần hỏi lại người dùng
    if "UNCLEAR" in refined_query:
        # Trường hợp câu hỏi không rõ ràng, hỏi lại người dùng
        clarify_query_chain = (
            clarify_query_prompt  # Sử dụng prompt để hỏi lại câu hỏi
            | chatbot.llm_gemini  # Sử dụng Large Language Model để sinh output
            | StrOutputParser()  # Parser để chuyển đổi output về dạng string
        )
        refined_query = clarify_query_chain.invoke({"chat_history": chatbot.history_conversation, "unclear_query": refined_query})
        logger.info(f"\nCâu hỏi sau khi hỏi lại: {refined_query}")
    
    # # Kiểm tra trường hợp câu hỏi không liên quan - xuất ra thông báo hỏi người dùng và chuyển hướng về du lịch - không thêm vào lịch sử hội thoại
    # if "NON_TRAVEL" in refined_query:
    #     logger.info(f"\nCâu hỏi không liên quan: {refined_query}")
    # elif "ASK" in refined_query:
    #     chatbot.history_conversation.append({"role": "user", "content": remove_tags(refined_query)})
    #     chatbot.history_conversation.append({"role": "assistant", "content": remove_tags(refined_query)})
    # else:
    #     # Thêm câu hỏi vào lịch sử hội thoại
    #     chatbot.history_conversation.append({"role": "user", "content": remove_tags(refined_query)})
        
    logger.info(f"Lịch sử hội thoại: {chatbot.history_conversation}\n")
    
    if "ASK" in refined_query or "NON_TRAVEL" in refined_query:
        return remove_tags(refined_query)
    
    return handle_next_steps(chatbot, remove_tags(refined_query))

def query_history(chatbot, query):
    """Truy vấn thông tin từ lịch sử hội thoại."""
    query_history_chain = (
        query_history_prompt_template
        | chatbot.llm_gemini
        | StrOutputParser()
    )
    response = query_history_chain.invoke({"history": chatbot.history_conversation, "query": query})
    return response

def summary_history(chatbot, conversation):
    """Tóm tắt lịch sử hội thoại."""
    summary_history_chain = (
        summary_history_prompt_template
        | chatbot.llm_gemini
        | StrOutputParser()
    )
    response = summary_history_chain.invoke({"history": chatbot.history_conversation, "new_conversation": conversation})
    return response

def handle_next_steps(chatbot, query):
    """Xử lý các bước tiếp theo sau khi tinh chỉnh truy vấn."""
    response = chatbot.agent_answer_travel_executor.invoke({"input": query})
    return response

def generate_response(chatbot, user_input, prompt_template_for_query):
    """Sinh phản hồi dựa trên truy vấn và ngữ cảnh."""
    logger.info(f"\nSinh ra câu trả lời từ câu truy vấn: {user_input}\n")
    cleaned_query_generation = RunnableLambda(lambda x: chatbot.query_generation_chain.invoke({"question": x}).get("text", ""))
    
    retrieval_chain_rag_fusion = (
        cleaned_query_generation
        | chatbot.retriever.map()
        | RunnableLambda(lambda results: reciprocal_rank_fusion(results))
    )
    
    def format_input(inputs):
        context_result = retrieval_chain_rag_fusion.invoke(inputs)
        return {"retrieved_context": context_result, "question": inputs["question"]}
    
    formatted_prompt = RunnableLambda(lambda x: prompt_template_for_query.format(**x))
    
    final_rag_chain = (
        RunnableLambda(format_input)
        | formatted_prompt
        | chatbot.llm_gemini
        | StrOutputParser()
    )
    
    return final_rag_chain.invoke({"question": user_input})

### Các hàm xử lý khi gọi các tool cho 4 chủ đề
def location_info_function(chatbot, query):
    print("Nhánh location_info_function")
    response = generate_response(chatbot= chatbot, user_input= query, prompt_template_for_query= location_info_prompt)
    return response

def itinerary_planner_function(chatbot, query):
    print("Nhánh itinerary_planner_function")
    response = generate_response(chatbot= chatbot, user_input= query, prompt_template_for_query= itinerary_planner_prompt)
    return response

def weather_info_function(chatbot, query):
    print("Nhánh weather_info_function")
    response = generate_response(chatbot= chatbot, user_input= query, prompt_template_for_query= weather_info_prompt)
    return response

def travel_faq_function(chatbot, query):
    print("Nhánh travel_faq_function")
    response = generate_response(chatbot= chatbot, user_input= query, prompt_template_for_query= travel_faq_prompt)
    return response