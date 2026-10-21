from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

#1. 创建对象
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

#2. 调用方法
# res = llm.invoke(input=[
#     {
#         "role":"system",
#         "content":"你是一名尚硅谷大模型讲师"
#     },
#     {
#         "role":"user",
#         "content":"你是谁?"
#     }
# ])
import logging
logging.basicConfig(level=logging.DEBUG)
res = llm.invoke(
    input=[
        SystemMessage("你是一名尚硅谷大模型讲师"),
        HumanMessage("你是谁?")
    ]
)
print(type(res))
print(res)