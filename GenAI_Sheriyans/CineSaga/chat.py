import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model = "mistral-small-2506",temperature=0.9)

response = model.invoke("write a poem on AI")

print(response.content)