"""03 · Reducer：并行分支写同一个 key 时，谁来合并？

考点：Annotated[list, operator.add]、自定义 reducer、InvalidUpdateError
思路：两个并行节点都往上写同一个 key，默认通道只允许「每步一个值」，
      所以必须用 reducer 声明合并规则：追加、覆盖、取最大……语义由 reducer 决定
观察：不加 reducer 会直接报 InvalidUpdateError；换成 keep_last 就只剩一条
运行：python 5_langgraph/03_reducer.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import StateGraph, START, END


#1. 追加型 reducer：把每步的 list 拼起来（最常用，日志/消息/中间结果都靠它）
class AppendState(TypedDict):
    aggregate: Annotated[list, operator.add]


#2. 覆盖型 reducer：只保留最后一次写入（"最新值"语义）
def keep_last(old_value, new_value):
    return new_value


class ReplaceState(TypedDict):
    aggregate: Annotated[list, keep_last]


#3. 完全不加 reducer：默认通道（同一超步只能写一次）
class PlainState(TypedDict):
    aggregate: list


def node_a(state):
    return {"aggregate": ["A"]}


def node_b(state):
    return {"aggregate": ["B"]}


def build(state_schema):
    builder = StateGraph(state_schema=state_schema)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    # 两条边从 START 扇出，两个节点在同一超步里并行跑
    builder.add_edge(START, "node_a")
    builder.add_edge(START, "node_b")
    builder.add_edge("node_a", END)
    builder.add_edge("node_b", END)
    return builder.compile()


def main():
    print("① operator.add（追加）：", build(AppendState).invoke({}))

    print("② 自定义 keep_last（覆盖）：", build(ReplaceState).invoke({}))

    # ③ 不加 reducer：两个并行节点写同一个普通通道，直接报错
    try:
        build(PlainState).invoke({})
    except InvalidUpdateError as exc:
        print("③ 不加 reducer：", type(exc).__name__, "->", str(exc).splitlines()[0])

    print(
        "\n结论：reducer 不是可选项而是合并协议。"
        "\n      · 并行写同一 key → 必须 Annotated 声明合并方式，否则 InvalidUpdateError"
        "\n      · 语义是追加还是覆盖，由业务决定，别为了不报错一律 operator.add"
    )


if __name__ == "__main__":
    main()
