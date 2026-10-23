"""04 · Checkpointer：让图记住上一轮说了什么

考点：checkpointer、thread_id、get_state / get_state_history、InMemorySaver 与 SqliteSaver
思路：图本身无状态，状态存在 checkpointer 里；同一 thread_id 就是同一段会话，
      换 thread_id 就是另一个用户，历史不串
观察：同一 thread 第二轮能看到第一轮的历史；换 thread 从零开始；history 条数 = 步数
运行：python 5_langgraph/04_checkpointer.py
"""

import operator
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph


class ChatState(TypedDict):
    question: str
    answer: str
    history: Annotated[list, operator.add]  # 每轮把问题追加进来，模拟"记忆"


def chat_node(state: ChatState):
    # 这里的 state["history"] 已经包含之前轮次的内容（由 checkpointer 恢复）
    history = state.get("history", [])
    turn = len(history) + 1
    return {
        "history": [state["question"]],
        "answer": f"第 {turn} 轮｜历史共 {len(history)} 条｜本轮问题：{state['question']}",
    }


def build_graph(checkpointer):
    builder = StateGraph(state_schema=ChatState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


def main():
    #1. SQLite checkpointer：落盘，进程重启后仍能续上（长期记忆）
    db_path = Path(__file__).with_name("checkpoint.db")
    db_path.unlink(missing_ok=True)  # 每次运行从干净状态开始
    conn = sqlite3.connect(database=str(db_path), check_same_thread=False)
    graph = build_graph(SqliteSaver(conn=conn))

    thread_u1 = {"configurable": {"thread_id": "u1"}}

    #2. 同一 thread 连跑两轮：第二轮能读到第一轮写的 history
    print("u1 第一轮：", graph.invoke({"question": "我叫李俊飞"}, thread_u1)["answer"])
    print("u1 第二轮：", graph.invoke({"question": "我叫什么？"}, thread_u1)["answer"])

    #3. 换 thread_id：另一个用户，历史应是空的
    thread_u2 = {"configurable": {"thread_id": "u2"}}
    print("u2 第一轮：", graph.invoke({"question": "我叫什么？"}, thread_u2)["answer"])

    #4. 查看当前状态：next 表示"下一步要执行哪个节点"
    snapshot = graph.get_state(thread_u1)
    print("\nget_state.u1.next =", snapshot.next, "｜values.history =", snapshot.values["history"])

    #5. 查看历史：每一超步一条快照，可以逐个回放（时间旅行）
    history = list(graph.get_state_history(thread_u1))
    print("get_state_history.u1 条数 =", len(history))
    for snap in history[:3]:
        print("   step", snap.metadata.get("step"), "| source", snap.metadata.get("source"),
              "| history", snap.values.get("history"))

    conn.close()

    # 说明：InMemorySaver 用法完全一样，只是数据在内存里，进程结束就没了：
    #   from langgraph.checkpoint.memory import InMemorySaver
    #   graph = build_graph(InMemorySaver())


if __name__ == "__main__":
    main()
