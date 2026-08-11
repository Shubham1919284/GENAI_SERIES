from dotenv import load_dotenv
load_dotenv('./notebooks/.env')
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# Setup DB
db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "history" not in st.session_state:
    st.session_state.history = []

model = ChatOllama(model="llama3.2:3b") 

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

TASK RULES:
    1. limit SELECT queries to 10 results with ORDER BY created_at DESC
    2. After CREATE/UPDATE/DELETE, confirm with the SELECT query
    3. IMPORTANT FORMATTING: ALWAYS present list/SQL outputs as a clean, structured Markdown Table (`| id | title | description | status | created_at |`). NEVER output messy raw SQL tables like `---+----------------+`. Convert truthy column values (e.g. 'f', 't') into readable words like 'false' or 'true'.

CRUD OPERATIONS:
    1. CREATE: Add a new task with title, description, and status
    2. READ: SELECT * FROM tasks WHERE ...LIMIT 10
    3. UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    4. DELETE: DELETE FROM tasks WHERE id=? OR title=?    

Table schema: id,title,description,status(pending/in_progress/completed),created_at
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=st.session_state.memory
)

## Building the UI
st.subheader("Simple Local Chatbot")

# Render chat history
for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Say something...")
if query:
    st.chat_message("user").markdown(query)

    # Convert chat history + new query into Langchain message objects
    formatted_messages = []
    for msg in st.session_state.history:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        else:
            formatted_messages.append(AIMessage(content=msg["content"]))
    formatted_messages.append(HumanMessage(content=query))

    try:
        # Perform Agent generation
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}}
        )
        message = response["messages"][-1].content

        with st.chat_message("ai"):
            st.markdown(message)

        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({"role": "ai", "content": message})
        
    except Exception as e:
        st.error(f"Error: {e}")
