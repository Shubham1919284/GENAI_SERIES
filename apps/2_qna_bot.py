from dotenv import load_dotenv
load_dotenv('./notebooks/.env')  # ✅ Correct path
import os

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "history" not in st.session_state:
    st.session_state.history = []

GROQ_API = os.getenv("GROQ_API")
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API, streaming=True)
search = GoogleSerperAPIWrapper()

@tool
def google_search(query: str) -> str:
    """Search Google for current, real-time information on any topic."""
    return search.run(query)

agent = create_agent(
    model=llm,
    tools=[google_search],
    checkpointer=st.session_state.memory,
    system_prompt="You are a helpful assistant that can search Google for current information.")


## Building the UI
st.subheader("QuickAnswer - Answer at the speed of thoughts")

for message in st.session_state.history:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)

query=st.chat_input("Ask a question")
if query:
    st.chat_message("user").markdown(query)

    response=agent.stream(
    {"messages":[{"role":"user", "content":query}]},
    {"configurable": {"thread_id": "1"}},
    stream_mode="messages"
    )

    ai_container=st.chat_message("ai")
    with ai_container:
        space=st.empty()

        message=""

        for chunk in response:
            message=message+chunk[0].content
            space.write(message)

    # st.session_state.history.append({"role":"user", "content":query})
    st.session_state.history.append({"role":"ai", "content":message})




