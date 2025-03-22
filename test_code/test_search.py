from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key="AIzaSyA-MAlE62P8Gg2g664zwnYcRAtNykEg_tE"
)

# tools = [TavilySearchResults(max_results=5)]
# agent = create_react_agent(llm, tools, prompt=hub.pull("hwchase17/react"))
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# print(agent_executor.invoke({"input": "Thời tiết ở Hồ Chí Minh ngày 21/3/2025"}))

search = TavilySearchResults(max_results=2)
search_results = search.invoke("Thời tiết ở Hồ Chí Minh ngày 21/3/2025")
print(search_results)