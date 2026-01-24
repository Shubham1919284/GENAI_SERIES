from dotenv import load_dotenv
import os
load_dotenv('./notebooks/.env')  # ✅ Correct path

import streamlit as st
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_GPT_API_KEY"), 
    base_url="https://openrouter.ai/api/v1",     
    max_tokens=1000
)

st.title("🤖 AskBuddy - AI Qna Bot")
st.markdown("My Qna Chatbot with LangChain and OpenAI - GPT !!")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    role=message['role']
    content=message['content']
    st.chat_message(role).markdown(content)

query=st.chat_input("Ask anything?")
if query:
    st.session_state.messages.append({"role":"user", "content":query})
    st.chat_message("user").markdown(query)
    
    res=llm.invoke(st.session_state.messages)  # ✅ Uses full conversation history
    st.chat_message("ai").markdown(res.content)
    
    st.session_state.messages.append({"role":"ai", "content":res.content})