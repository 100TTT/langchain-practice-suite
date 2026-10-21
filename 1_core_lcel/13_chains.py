import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def prompt_template_demo(product: str, aspect1: str, aspect2: str) -> None:
    logging.basicConfig(level=logging.DEBUG)
    # 定义提示词模板 - 必须使用元组格式 ("role", "模板字符串") 才能进行变量替换
    chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的评论员"),
        ("human", "请评价{product}的优缺点，包括{aspect1}和{aspect2}两方面。"),
    ])

    # 定义大模型
    llm = ChatOpenAI(
        model="gpt-4o-mini"
    )

    # 定义解析器对象
    str_output_parser = StrOutputParser()

    # 组成调用链
    chain = chat_prompt_template | llm | str_output_parser

    # chain = RunnableSequence(chat_prompt_template, llm ,str_output_parser)

    res = chain.invoke({
        "product": product,
        "aspect1": aspect1,
        "aspect2": aspect2,
    })
    """for chunk in chain.stream({"product": "iphone15", "aspect1": "外观", "aspect2": "性能"}):
        print(chunk, end="", flush=True)"""

    print(res)
if __name__ == "__main__":
    prompt_template_demo("iphone15","外观","性能")
