import asyncio
import logging

from langchain_openai import ChatOpenAI
import time
#1. 模拟发送三个请求
MESSAGES = [
    "什么是Langchain?",
    "什么是RAG?",
    "什么是Agent?"
]

def call_llm_sync(messages:list[str]):
    # 2. 创建对象
    llm = ChatOpenAI(
        model="gpt-4o-mini"
    )
    # 记录时间
    time0 = time.perf_counter()
    for message in messages:
        llm.invoke(message)
    time1 = time.perf_counter()
    print(f"同步调用耗时为:{time1-time0}")

async def call_llm_async(messages: list[str]):
    logging.basicConfig(level=logging.DEBUG)
    # 2. 创建对象
    llm = ChatOpenAI(
        model="gpt-4o-mini"
    )
    # 记录时间 - 使用 asyncio.gather 并行执行多个请求
    time0 = time.perf_counter()
    tasks = [llm.ainvoke(msg) for msg in messages]
    #并发执行
    await asyncio.gather(*tasks)
    time1 = time.perf_counter()
    print(f"异步并行调用耗时为:{time1 - time0}")

if __name__ == "__main__":
    # call_llm_sync(MESSAGES)
    asyncio.run(call_llm_async(MESSAGES))
