from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

#1. 创建agent
# 通过checkpointer让agent拥有记忆
agent = create_agent(
    model="gpt-4o-mini",
    checkpointer=InMemorySaver()
)

#2. 第一次调用模型
res = agent.invoke({
    "messages":[
        {"role":"user","content":"什么是Langchain?"}
    ]
},config={
        "configurable":{"thread_id":"1"}
    })

print(f"第一次调用大模型的返回结果:{res['messages'][-1].content}")

#2. 第二次调用模型
res2 = agent.invoke({
    "messages":[
        {"role":"user","content":"我刚刚问了你什么问题?"}
    ]
},config={
        "configurable":{"thread_id":"1"}
    })
# checkpoint_ns, checkpoint_id
print(f"第二次调用大模型的返回结果:{res2['messages'][-1].content}")