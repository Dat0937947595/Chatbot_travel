import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from scr.base.backend.rag.chain import answer_str
#from chain import answer_str

def get_response(query):
    # template = """
    # You are a helpful assistant. Answer the following questions considering the history of the conversation:

    # Chat history:{chat_history}

    # User question:{user_question}
    # """

    # return chain.invoke({
    #     "chat_history": chat_history,
    #     "user_question": query
    #     }
    # )

    #generate
    # return chain.stream({
    #     "chat_history": chat_history,
    #     "user_question": query
    #     }
    # )
    answer = answer_str(query)
    print(answer)
    return answer

def run_streamlit():
    st.set_page_config(page_title="Streaming Bot", page_icon="Z")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Streaming Bot")
    #Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.markdown(message.content)
        else:
            with st.chat_message("ai"):
                st.markdown(message.content)
        
    user_query = st.chat_input("Your message")
    print(user_query)
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(user_query))

        with st.chat_message("human"):
            st.markdown(user_query)
        
        with st.chat_message("ai"):
            #ai_response = get_response(user_query, st.session_state.chat_history)
            #st.markdown(ai_response)
            #generate
            ai_response = get_response(user_query)
            # ai_response = "i don't know"
            # print(ai_response)
            print(user_query)
            print("2")
            st.markdown(ai_response)
            print("1")
        st.session_state.chat_history.append(AIMessage(ai_response))
