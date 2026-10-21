import logging

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
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

#1. 创建JsonOutputParser对象
json_output_parser = JsonOutputParser(pydantic_object=Prime)
logging.basicConfig(level=logging.DEBUG)
res = llm.invoke(
    input=[
        {
            "role":"system",
            "content":json_output_parser.get_format_instructions()
        },
        {
            "role":"user",
            "content":"任意生成5个1000-100000之间素数，并标出小于该素数的素数个数"
        }
    ]
)
print(res.content)
# 使用json_output_parser将输出的json字符串转换成json对象
json_obj = json_output_parser.invoke(res)
print(json_obj['prime'])
print(json_obj['count'])

