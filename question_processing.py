'''
	Dùng đề xử lý các vấn đề liên quan tới câu hỏi người dùng:
		- Viết lại câu hỏi (nếu cần thiết)
		- Hỏi lại câu hỏi (nếu cần thiết)
		- Loại các câu hỏi không liên quan
		- ...
'''
### Import thư viện
import logging
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
from prompt_templates import *
from langchain_core.output_parsers import StrOutputParser

''' -------------------------------------- '''

### Load các biến môi trường
dotenv.load_dotenv()

''' -------------------------------------- '''

### Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Question Processing")

''' -------------------------------------- '''

### Hàm refine query dựa trên lịch sử hội thoại
def refine_query(user_query, chat_history):
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

if __name__ == "__main__":
	# Câu hỏi gốc
	llm = ChatGoogleGenerativeAI(
		model="gemini-2.0-flash",
	)

	# from langchain_groq import ChatGroq
	# llm = ChatGroq(
	# 	model_name="deepseek-r1-distill-llama-70b",
	# 	api_key="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"
	# )
	logger.info("Khởi tạo mô hình gemini-2.0-flash thành công.")
	chat_history = []

	''' Test refine query '''
	# Refine câu hỏi
	while True:
		user_query = input("Nhập câu hỏi của bạn: ")	
		refined_query = refine_query(user_query, chat_history)
		logger.info(f"\nCâu hỏi sau refine: {refined_query}")

		# Kiểm tra trường hợp câu hỏi không rõ ràng - nếu có cần hỏi lại người dùng
		if "UNCLEAR" in refined_query:
			# Trường hợp câu hỏi không rõ ràng, hỏi lại người dùng
			clarify_query_chain = (
				clarify_query_prompt  # Sử dụng prompt để hỏi lại câu hỏi
				| llm  # Sử dụng Large Language Model để sinh output
				| StrOutputParser()  # Parser để chuyển đổi output về dạng string
			)
			refined_query = clarify_query_chain.invoke({"chat_history": chat_history, "unclear_query": refined_query})
			logger.info(f"\nCâu hỏi sau khi hỏi lại: {refined_query}")
		
		# Kiểm tra trường hợp câu hỏi không liên quan - xuất ra thông báo hỏi người dùng và chuyển hướng về du lịch - không thêm vào lịch sử hội thoại
		if "NON_TRAVEL" in refined_query:
			logger.info(f"\nCâu hỏi không liên quan: {refined_query}")
		else:
			# Thêm câu hỏi vào lịch sử hội thoại
			chat_history.append({"role": "user", "content": user_query})
			chat_history.append({"role": "assistant", "content": refined_query})	
		logger.info(f"Lịch sử hội thoại: {chat_history}\n")