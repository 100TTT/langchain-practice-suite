"""08 · 节点重试：模型抖一下不该整条工作流失败

考点：RetryPolicy(max_attempts=...)、重试边界、节点幂等
思路：给"会抖"的节点（LLM 调用、外部 API）挂重试策略，次数用尽才抛错；
      重试发生在节点级，不需要自己在代码里写 while + sleep
观察：前几次抛异常被自动重试，最后一次成功，图正常返回
运行：python 5_langgraph/08_node_retry.py
"""

from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy


class State(TypedDict):
    query: str
    result: str


attempt = {"n": 0}


def flaky_llm_node(state: State):
    """模拟不稳定的模型接口：前 3 次失败，第 4 次成功。"""
    attempt["n"] += 1
    if attempt["n"] <= 3:
        print(f"    第 {attempt['n']} 次调用失败（模拟超时）")
        raise ConnectionError("upstream timeout")
    print(f"    第 {attempt['n']} 次调用成功")
    return {"result": f"(LLM) 关于「{state['query']}」的回答"}


def main():
    builder = StateGraph(state_schema=State)
    # max_attempts=5：总共最多试 5 次；只加在这个节点上，别给全图加
    builder.add_node("llm", flaky_llm_node, retry_policy=RetryPolicy(max_attempts=5))
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)

    graph = builder.compile()
    result = graph.invoke({"query": "什么是 LangGraph？"})
    print("\n结果：", result)
    print("实际尝试次数：", attempt["n"])

    print(
        "\n要点："
        "\n  · 重试是节点级策略，只包住「会抖」的那一步，不会把整张图重跑"
        "\n  · 次数用尽仍未成功会抛出异常，交给上层（人审 / 降级 / 报错）"
        "\n  · 被重试的节点必须幂等：重试可能已经产生副作用（写库、扣费）"
        "\n  · 想只对特定异常重试，可用 RetryPolicy(retry_on=(TimeoutError,))"
    )


if __name__ == "__main__":
    main()
