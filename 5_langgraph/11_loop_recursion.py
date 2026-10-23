"""11 · 循环与 recursion_limit：自己转的圈，自己刹住

考点：条件边构成环、终止条件、recursion_limit（限制超步数）
思路：Agent 的"想 → 做 → 看结果 → 再想"天生是环。环不可怕，没有终止条件才可怕；
      recursion_limit 是最后一道保险，防止逻辑写错时无限循环
观察：① 正常终止的循环跑 3 轮就结束；② 限制设小会抛 GraphRecursionError
运行：python 5_langgraph/11_loop_recursion.py
"""

from typing import Literal, TypedDict

from langgraph.constants import START, END
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph


class LoopState(TypedDict):
    count: int
    max_count: int
    result: str


def draft(state: LoopState):
    """每轮把计数 +1，相当于 Agent 的一次"再试一次"。"""
    count = state["count"] + 1
    print(f"  draft 第 {count} 轮")
    return {"count": count, "result": f"第 {count} 稿"}


def should_continue(state: LoopState) -> Literal["improve", "__end__"]:
    # 终止条件必须显式写：达到目标就结束，否则继续改进
    if state["count"] >= state["max_count"]:
        return END
    return "improve"


def improve(state: LoopState):
    print(f"  improve（基于「{state['result']}」继续打磨）")
    return {"result": state["result"] + "+"}


def build_graph():
    builder = StateGraph(state_schema=LoopState)
    builder.add_node("draft", draft)
    builder.add_node("improve", improve)

    builder.add_edge(START, "draft")
    # 条件边：满足条件就 END，否则回到 improve，再由 improve 回到 draft —— 成环
    builder.add_conditional_edges("draft", should_continue, ["improve", END])
    builder.add_edge("improve", "draft")
    return builder.compile()


def main():
    graph = build_graph()

    print("① 有终止条件的环（max_count=3）")
    result = graph.invoke({"count": 0, "max_count": 3, "result": ""})
    print("   结果：", result, "\n")

    print("② 终止条件写错时，recursion_limit 兜底（max_count=100，限制 8 步）")
    try:
        graph.invoke(
            {"count": 0, "max_count": 100, "result": ""},
            {"recursion_limit": 8},  # 一轮循环 = draft + improve 两个超步
        )
    except GraphRecursionError as exc:
        print("   捕获：", str(exc).splitlines()[0])

    print(
        "\n要点："
        "\n  · recursion_limit 数的是超步（superstep），不是节点数：一轮循环通常占 2 步"
        "\n  · 生产上要同时做：显式终止条件 + 最大轮数上限 + recursion_limit 兜底"
        "\n  · 循环里累积的状态建议用 reducer（见 03），否则每轮会互相覆盖"
    )


if __name__ == "__main__":
    main()
