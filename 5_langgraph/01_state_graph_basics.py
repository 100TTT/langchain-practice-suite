"""01 · StateGraph 七步：状态 → 节点 → builder → 加节点 → 加边 → 编译 → invoke

考点：TypedDict 状态、START / END、input_schema 与 output_schema、扇出与扇入
思路：把「同时查 RAG 和 Web、再交给 LLM 汇总」写成一张最小图，先把七步跑顺，
      后面 02-11 都是在这张图上逐项加能力
观察：invoke 只传 input_schema 里的字段；返回值被 output_schema 裁剪成 {final_answer}
运行：python 5_langgraph/01_state_graph_basics.py
"""

from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


#1. 定义状态：整张图共享的一块「黑板」，节点往里写、也从里面读
class MyState(TypedDict):
    input: str
    web_search_result: str
    rag_search_result: str
    final_answer: str


class MyInput(TypedDict):
    """允许从外部传进来的字段（不写死成 MyState，调用方就不能乱传）"""
    input: str


class MyOutput(TypedDict):
    """最终返回给调用方的字段（中间结果不泄漏，接口更干净）"""
    final_answer: str


#2. 定义节点：入参是当前状态，返回 dict 只写自己要更新的字段
def rag_search_node(state: MyState):
    return {"rag_search_result": f"[RAG] 针对「{state['input']}」召回 3 条文档片段"}


def web_search_node(state: MyState):
    return {"web_search_result": f"[Web] 针对「{state['input']}」拿到 5 条实时结果"}


def llm_invoke_node(state: MyState):
    # 注意：llm_invoke 有两条入边，两条上游各自完成时都会触发它一次（超步机制），
    # 所以这里用 .get() 兜底——两路都要等齐再汇总的写法见 05_superstep_parallel.py
    rag = state.get("rag_search_result", "（还没到）")
    web = state.get("web_search_result", "（还没到）")
    return {"final_answer": f"（模拟 LLM 汇总）问题={state['input']}｜RAG={rag}｜Web={web}"}


def main():
    #3. builder：一次性指定状态、输入、输出三个 schema
    builder = StateGraph(state_schema=MyState, input_schema=MyInput, output_schema=MyOutput)

    #4. 加节点
    builder.add_node("rag_search", rag_search_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("llm_invoke", llm_invoke_node)

    #5. 加边：START 同时指向两个检索节点（扇出），两个检索节点都指向 llm（扇入）
    builder.add_edge(START, "rag_search")
    builder.add_edge(START, "web_search")
    builder.add_edge("rag_search", "llm_invoke")
    builder.add_edge("web_search", "llm_invoke")
    builder.add_edge("llm_invoke", END)

    #6. 编译：编译后才是可执行对象，前面都只是「画布」
    graph = builder.compile()

    #7. 执行：传入的键必须是 MyInput 的字段，返回的键被 MyOutput 裁剪
    print("节点集合：", list(graph.nodes))
    print("执行结果：", graph.invoke({"input": "如何使用 LangGraph"}))


if __name__ == "__main__":
    main()
