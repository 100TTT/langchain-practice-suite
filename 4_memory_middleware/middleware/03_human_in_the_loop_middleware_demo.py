#1. 编写一个工具: 用于转账
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 创建一个HumanInTheLoopMiddleware，要求用户确认后才执行转账操作
human_in_the_loop_middleware = HumanInTheLoopMiddleware(
    interrupt_on={"transfer_money":True,"get_weather":False}
)

@tool(description="查询某个城市在指定日期下的天气信息")
def get_weather(city:str,date:str):
    # 这里我们模拟一个天气信息的返回
    return f"{city}在{date}的天气是晴朗，温度25度。"

@tool(description="一个转账工具方法，执行将amount转给to_account")
def transfer_money(amount: int, to_account: str) -> str:
    """转账工具 (敏感操作)"""
    print(f"!!! 正在执行转账: {amount} -> {to_account} !!!")
    return f"成功转账 {amount} 元给 {to_account}"

#2. 创建Agent调用工具
agent = create_agent(
    model="gpt-4o-mini",
    tools=[transfer_money,get_weather],
    middleware=[human_in_the_loop_middleware],
    checkpointer=InMemorySaver()
)

#3. 调用大模型
res1 = agent.invoke(
    {
        "messages":[{"role":"user","content":"向张三转账500元"}]
    },
    config={"configurable":{"thread_id":"1"}}
)
# 获取到打断的信息
interrupt = res1['__interrupt__']
print(interrupt)

# 进行人工确认后再次调用大模型
command = Command(resume={"decisions":[{"type":"edit","edited_action":{"name":"transfer_money","args":{"amount":50,"to_account":"张三丰"}}}]})
res_command = agent.invoke(command,config={"configurable":{"thread_id":"1"}})

print(res_command)

res2 = agent.invoke(
    {
        "messages":[{"role":"user","content":"查询北京2026年3月23号的天气情况"}]
    },
    config={"configurable":{"thread_id":"2"}}
)
print(res2)
