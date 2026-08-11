from dotenv import load_dotenv
# load_dotenv()
import os
load_dotenv('./notebooks/.env')  # ✅ Correct path


from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

GROQ_API=os.getenv("GROQ_API")
model = ChatGroq(model="openai/gpt-oss-20b", api_key=GROQ_API)
search = GoogleSerperAPIWrapper()
memory = MemorySaver()


agent = create_agent(
    model=model,
    tools=[search.run],
    checkpointer=memory,
    system_prompt="You are a agent and can search for any question on google."
)


while True:
    query = input("User: ")
    if query.lower() == "quit":
        print("Good Bye 👋")
        break

    response = agent.invoke(
                {"messages":[{"role":"user", "content":query}]}, 
                {"configurable": {"thread_id": "1"}}, 
            )
    print("AI:",  response["messages"][-1].content)