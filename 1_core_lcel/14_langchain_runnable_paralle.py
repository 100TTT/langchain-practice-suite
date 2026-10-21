import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini"
)
logging.basicConfig(level=logging.DEBUG)
#1. 创建第一条调用链
# 1.1 定义提示词模板
english_prompt = ChatPromptTemplate([("human", "把这个句子{topic}翻译成英文")])
# 1.2 定义结果解析器
str_output_parser = StrOutputParser()
english_chain = english_prompt | llm | str_output_parser

#2. 第二条调用链
# 2.1 定义提示词模板
Korean_prompt = ChatPromptTemplate([("human", "把这个句子{topic}翻译成韩文")])
korean_chain = Korean_prompt | llm | str_output_parser

#3. 并行调用这两条链
chain_map = RunnableParallel(
    english = english_chain,
    korean = korean_chain
)

res = chain_map.invoke({
    "topic":"你好，我是科比"
})

print(res)
