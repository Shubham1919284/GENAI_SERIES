from dotenv import load_dotenv
load_dotenv('./notebooks/.env')
import os

from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# Setup DB
db = SQLDatabase.from_uri("sqlite:///my_tasks.db")
db.run("""
    CREATE TABLE IF NOT EXISTS tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK(status IN ('pending','in_progress','completed')) DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
""")


# GROQ_API = os.getenv("GROQ_API")
model = ChatOllama(model="llama3.2:3b") 
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
memory = MemorySaver()

system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

TASK RULES:
    1. limit SELECT queries to 10 results with ORDER BY created_at DESC
    2. After CREATE/UPDATE/DELETE, confirm with the SELECT query
    3. If the user request to a list of tasks, present the output in a structured format to ensure a clean and organiszed display in the browser.

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
    checkpointer=memory
)

while True:
    query = input("user: ")
    if query.lower() in ['quit', 'exit', 'q']:
        break
        
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}}
        )
        result = response['messages'][-1].content
        print("AI:", result)
    except Exception as e:
        if "rate_limit_exceeded" in str(e):
            print("AI: [Error] Groq API Rate Limit Exceeded. Please wait a moment before trying again.")
        else:
            print(f"AI: [Error] An unexpected error occurred: {e}")
