"""02 · 私有状态：每个节点只声明自己用到的字段

考点：节点级 TypedDict（私有输入 schema）
思路：01 里每个节点都拿完整的 MyState，节点一多就分不清"谁读谁写"；
      给每个节点单独声明它关心的字段，读代码时一眼能看出依赖
观察：节点只读写自己的子集，返回值依然合并进全局状态
运行：python 5_langgraph/02_private_state.py
"""

from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


#1. 全局状态：整张图的公共黑板
class MyState(TypedDict):
    input: str
    web_search_result: str
    rag_search_result: str
    final_answer: str


#2. 节点私有状态：只包含该节点真正用到的字段
class WebSearchState(TypedDict):
    input: str
    web_search_result: str


class RagSearchState(TypedDict):
    input: str
    rag_search_result: str


#3. 节点实现：签名写子集，代码里就只会碰子集字段
def rag_search_node(state: RagSearchState):
    return {"rag_search_result": f"[RAG] 针对「{state['input']}」召回 3 条文档片段"}


def web_search_node(state: WebSearchState):
    return {"web_search_result": f"[Web] 针对「{state['input']}」拿到 5 条实时结果"}


def llm_invoke_node(state: MyState):
    # 汇总节点需要两路结果，所以它声明完整的 MyState
    return {
        "final_answer": (
            f"（模拟 LLM 汇总）问题={state['input']}"
            f"｜RAG={state['rag_search_result']}"
            f"｜Web={state['web_search_result']}"
        )
    }


def main():
    builder = StateGraph(state_schema=MyState)

    builder.add_node("rag_search", rag_search_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("llm_invoke", llm_invoke_node)

    builder.add_edge(START, "rag_search")
    builder.add_edge(START, "web_search")
    builder.add_edge("rag_search", "llm_invoke")
    builder.add_edge("web_search", "llm_invoke")
    builder.add_edge("llm_invoke", END)

    graph = builder.compile()
    # 这次只传 input，web_search_result 由节点自己填——01 里为了不 KeyError 手动补了它
    print("执行结果：", graph.invoke({"input": "如何使用 LangGraph"}))

    # 说明：TypedDict 的字段标注是给人和 IDE 看的，运行时仍按全局 schema 合并；
    # 真正起约束作用的是 input_schema / output_schema（见 01）。
    print("\n提示：私有状态的价值是可读性与复用性，不是运行时校验。")


if __name__ == "__main__":
    main()
