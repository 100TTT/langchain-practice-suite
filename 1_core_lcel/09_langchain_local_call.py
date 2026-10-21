import logging

from langchain_ollama import ChatOllama
logging.basicConfig(level=logging.DEBUG)
llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434"
)

res = llm.invoke([
    {"role":"system","content":"你是一名尚硅谷讲师"},
    {"role":"user","content":"你是谁?"}
])

print(res)