import logging

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="gpt-4o-mini"
)
"""
{
    "prime":[1,2,3,4,5],
    "count":[1,2,3,4,5]
}
"""
class Prime(BaseModel):
    prime: list[int] = Field(description="生成的素数列表")
    count: list[int] = Field(description="小于该素数的素数个数")

logging.basicConfig(level=logging.DEBUG)

#2. 将大模型对象转成配置了结果解析的大模型对象
structured_llm = llm.with_structured_output(Prime)

res = structured_llm.invoke(
    input=[
        {
            "role":"user",
            "content":"任意生成5个1000-100000之间素数，并标出小于该素数的素数个数"
        }
    ]
)

print(res.prime)
print(res.count)
