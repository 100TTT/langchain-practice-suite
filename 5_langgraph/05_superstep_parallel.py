"""05 · 超步与并行：节点是 Actor，边是通道，波次推进

考点：超步（superstep）执行模型、并行分支、扇入（fan-in）什么时候只跑一次
思路：同一超步里的节点一起跑，各自的更新合并进状态后再进入下一超步；
      本文件用两个结构对比：等深菱形 vs 深浅不一
观察：等深菱形 D 只执行 1 次；分支深度不同时 D 会被触发 2 次
运行：python 5_langgraph/05_superstep_parallel.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class State(TypedDict):
    aggregate: Annotated[list, operator.add]


def make_node(name):
    def node(state: State):
        # 打印能看出"这一波还有谁"——同一超步里 state 是同一个快照
        print(f'  {name} 执行，看到 aggregate={state["aggregate"]}')
        return {"aggregate": [name]}
    return node


def build(nodes, edges):
    builder = StateGraph(State)
    for name in nodes:
        builder.add_node(name, make_node(name))
    for src, dst in edges:
        builder.add_edge(src, dst)
    return builder.compile()


def main():
    #1. 等深菱形：a → (b, c) → d，两条支路深度相同
    graph_even = build(
        nodes=["a", "b", "c", "d"],
        edges=[(START, "a"), ("a", "b"), ("a", "c"),
               ("b", "d"), ("c", "d"), ("d", END)],
    )
    print("① 等深菱形（b、c 同一超步到达 d）")
    print("   结果：", graph_even.invoke({}))
    print("   -> d 只执行 1 次，两路结果在同一快照里合并\n")

    #2. 深浅不一：b→b_2 只有 2 步，c→c_2→c_3 有 3 步
    graph_uneven = build(
        nodes=["a", "b", "b_2", "c", "c_2", "c_3", "d"],
        edges=[(START, "a"), ("a", "b"), ("a", "c"),
               ("b", "b_2"), ("b_2", "d"),
               ("c", "c_2"), ("c_2", "c_3"), ("c_3", "d"),
               ("d", END)],
    )
    print("② 深浅不一（B 支路先到 d）")
    result = graph_uneven.invoke({})
    print("   结果：", result)
    print("   -> d 被触发 2 次，aggregate 里有两条 d\n")

    #3. 通道一览：每个状态字段对应一个 channel，reducer 决定怎么合并
    print("③ channels：", {k: type(v).__name__ for k, v in graph_uneven.channels.items()})

    print(
        "\n结论："
        "\n  · 同一超步内的节点并行执行，波次之间串行；recursion_limit 限制的就是超步数"
        "\n  · 扇入只在「同一超步到达」时合并成一次；分支深度不同，下游会被多次触发"
        "\n  · 所以并行写同一 key 必须配 reducer（见 03），下游逻辑要幂等，或让分支等深"
    )


if __name__ == "__main__":
    main()
