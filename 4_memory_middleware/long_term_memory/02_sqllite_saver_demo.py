from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
#1. 创建agent
# pip install langgraph-checkpoint-sqlite

# 通过sql_lite数据库checkpointer让agent拥有长期记忆
connection = sqlite3.connect(database="agent_memory.db",check_same_thread=False) # 建立本地轻量级数据库
sqlite_saver = SqliteSaver(conn=connection) # conn= 连接数据库
agent = create_agent(
    model="gpt-4o-mini",
    checkpointer=sqllite_saver
)

#2. 第一次调用模型
# res = agent.invoke({
#     "messages":[
#         {"role":"user","content":"什么是Langchain?"}
#     ]
# },config={
#         "configurable":{"thread_id":"1"}
#     })
#
# print(f"第一次调用大模型的返回结果:{res['messages'][-1].content}")

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