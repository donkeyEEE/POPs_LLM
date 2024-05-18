from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import os
os.chdir('E:\学习\python\py_codbase\POPs_LLM')
os.environ['TAVILY_API_KEY']='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'
import streamlit as st

st.set_page_config(page_title="LangChain: Chat with search", page_icon="🦜")
st.title("🦜 LangChain: Chat with search")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}


if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.chat_message("user").write(prompt)
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        # st.stop()
    from PK_LLM_endfront.Langgraph.Graph import app
    inputs = {'question':prompt,'history':''}
    r = app.invoke(inputs)
    st.write(r['output'])
        