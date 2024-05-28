from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# 获取api key
import os
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ['TAVILY_API_KEY']='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'
import streamlit as st

st.set_page_config(page_title="EnvironmentGPT", page_icon="💬")
st.header("📖 EnvironmentGPT: Chat with environmental Knowlegde")

# 设置边框
# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
with st.sidebar:
    st.markdown(
        "## How to use \n"
        "   "
        "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"
        "2. Configure the services you need in the settings."
        "3. Ask a question about the document💬\n"
    )
    api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            #value=os.environ.get("OPENAI_API_KEY", None)
            #or st.session_state.get("OPENAI_API_KEY", ""), # 这个控件首次渲染时的文本值
        )
    if api_key_input == 'lyzNB':
        st.session_state["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", None)
    else:
        st.session_state["OPENAI_API_KEY"] = api_key_input
        
    st.markdown("---")
    st.markdown("# About")
    st.markdown("📖 EnvironmentGPT is a chatbot which is designed to answer professional questions in the field of environment.")
    st.markdown('---')
    # 是否进行检索增强RAG
    if_rag = st.radio(
        label="Whether use retrieval-augment (Retrieval Augmented Generation,RAG).",
        options=[':rainbow[RAG]','**no RAG**'],
        index=0,
    )
    
    st.session_state['if_rag'] = (if_rag == ':rainbow[RAG]')
    st.markdown('\n\n---')


# Chat message history that stores messages in Streamlit session state.
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
from langchain_openai import ChatOpenAI
def default_agent(prompt:str)->str:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.session_state["OPENAI_API_KEY"], streaming=True)
    tools = []
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
    )
    r = executor.invoke(prompt)
    return r['output']

if len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

if prompt := st.chat_input(placeholder="什么是PFAS"):
    st.chat_message("user").write(prompt)
    if st.session_state['if_rag']:
        from PK_LLM_endfront.Langgraph.Graph import app
        inputs = {'question':prompt,'history':''}
        r = app.invoke(inputs)
        st.write(r['output'])
        # st.write(a)
    else:
        r  = default_agent(prompt=prompt)
        st.write(r)


