"""07 · 节点缓存：同样的输入不必算第二遍

考点：CachePolicy(ttl=...)、InMemoryCache、缓存键的构成
思路：检索/嵌入这类节点又慢又"幂等"，给它加缓存；相同输入在 TTL 内直接命中，
      命中时不执行节点函数，也不占 LLM token
观察：第二次同样输入耗时骤降且看不到节点内打印；换输入、或超过 TTL 就重新执行
运行：python 5_langgraph/07_node_cache.py
"""

import time
from typing import TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import CachePolicy


class State(TypedDict):
    topic: str
    rag_result: str
    web_result: str
    answer: str


def rag_node(state: State):
    return {"rag_result": f"[RAG] {state['topic']} 的 3 条内部文档片段"}


def slow_web_node(state: State):
    # 模拟一次贵的调用：搜索 API / 大模型
    print(f"    ↳ web 节点真实执行（topic={state['topic']}）")
    time.sleep(1.5)
    return {"web_result": f"[Web] {state['topic']} 的 5 条实时结果"}


def answer_node(state: State):
    return {"answer": f"综合：{state['rag_result']} + {state['web_result']}"}


def build_graph(ttl: int):
    builder = StateGraph(state_schema=State)
    builder.add_node("rag", rag_node)
    # 只给慢节点加缓存策略；缓存随编译时的 cache 存储生效
    builder.add_node("web", slow_web_node, cache_policy=CachePolicy(ttl=ttl))
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "rag")
    builder.add_edge(START, "web")
    builder.add_edge("rag", "answer")
    builder.add_edge("web", "answer")
    builder.add_edge("answer", END)

    # 注意：缓存只需要 cache=，不需要 checkpointer（04 那个是"记忆"，别混）
    return builder.compile(cache=InMemoryCache())


def timed(graph, payload, label):
    start = time.perf_counter()
    graph.invoke(payload)
    cost = time.perf_counter() - start
    print(f"  {label:<22} 耗时 {cost:5.2f}s")
    return cost


def main():
    graph = build_graph(ttl=60)

    print("① TTL=60s，同样的输入连跑两次：")
    first = timed(graph, {"topic": "LangGraph 条件边"}, "第一次（缓存未命中）")
    second = timed(graph, {"topic": "LangGraph 条件边"}, "第二次（缓存命中）")

    print("\n② 换一个输入：缓存键变了，重新执行")
    timed(graph, {"topic": "LangGraph 人审"}, "换 topic")

    print("\n③ TTL=1s，过期后再跑：")
    short_ttl = build_graph(ttl=1)
    timed(short_ttl, {"topic": "TTL 演示"}, "第一次")
    timed(short_ttl, {"topic": "TTL 演示"}, "1s 内第二次")
    time.sleep(1.2)
    timed(short_ttl, {"topic": "TTL 演示"}, "过期后第三次")

    speedup = f"约 {first / second:.0f}x" if second > 0.01 else "命中后耗时接近 0"
    print(
        "\n要点："
        f"\n  · 加速 {speedup}，命中的节点连函数体都没进"
        "\n  · 缓存键 ≈ 节点名 + 该节点看到的输入；输入变、节点名变都会失效"
        "\n  · 只给「纯函数型」节点加缓存；有副作用（写库、发消息）的节点别加"
    )


if __name__ == "__main__":
    main()
