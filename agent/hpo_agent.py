import os
import dotenv
from langchain_openai import OpenAI

# create an openai llm
import openai
 
client = openai.OpenAI(
    api_key="sk-kME1WhLQURCa7qnEOfmVsA",  
    base_url="https://llmapi.paratera.com"
)
 
response = client.chat.completions.create(
    model="GLM-4.7",  # model to send to the proxy
    messages=[{"role": "user", "content": "hello"}],
)
print(response)