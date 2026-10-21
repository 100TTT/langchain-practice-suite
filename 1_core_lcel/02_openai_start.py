from openai import OpenAI
#目标: 向OpenAI大模型输入一句话，拿到OpenAI的输出，并且将输出结果打印
str = "什么是Langchain?"
#1. 创建OpenAI大模型对象
#获取api_key和base_url
from dotenv import load_dotenv
import os
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
load_dotenv()
llm = OpenAI(
    base_url=base_url,
    api_key=api_key
)

#2. 调用对象的方法创建对话，拿到对话结果
res = llm.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role":"system",
        "content":"你是一名尚硅谷大模型讲师"
    },
    {
        "role":"user",
        "content":str
    }]
)

#3. 打印结果
print(res.choices[0].message.content)