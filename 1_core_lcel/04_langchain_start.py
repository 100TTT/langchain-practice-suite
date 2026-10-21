#langchain统一的这套api, init_chat_model
from langchain.chat_models import init_chat_model
#1. 创建对象
llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai"
)

#2. 调用方法创建对话
res = llm.invoke("你是谁?")

print(res)