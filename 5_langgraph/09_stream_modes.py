"""09 · 流式输出：四种 stream_mode 分别给谁用

考点：stream_mode = values / updates / messages / custom
思路：前端要"打字机"、运维要"进度条"、调试要"每步 diff"——不同消费方要不同粒度，
      一张图可以同时用多种模式（stream_mode=["updates", "custom"]）
观察：values 给整份状态、updates 给增量、messages 给 token、custom 给业务自定义事件
运行：python 5_langgraph/09_stream_modes.py
"""

from typing import TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime


class State(TypedDict):
    question: str
    rag_result: str
    web_result: str
    answer: str


def get_llm():
    """离线演示用假模型；换成真模型只要改成：
    from langchain.chat_models import init_chat_model
    return init_chat_model("openai:gpt-4o-mini")   # 读 .env 里的 Key / Base URL
    """
    return FakeListChatModel(
        responses=["LangGraph 把工作流写成状态图：节点改状态，边决定下一步怎么走。"]
    )


def rag_node(state: State, runtime: Runtime):
    # custom 模式：把业务进度主动推给前端
    runtime.stream_writer({"stage": "retrieval", "detail": "正在检索知识库…"})
    return {"rag_result": "3 条文档片段"}


def web_node(state: State, runtime: Runtime):
    runtime.stream_writer({"stage": "retrieval", "detail": "正在检索网络…"})
    return {"web_result": "5 条实时结果"}


def answer_node(state: State):
    message = get_llm().invoke(state["question"])
    return {"answer": message.content}


def build_graph():
    builder = StateGraph(state_schema=State)
    builder.add_node("rag", rag_node)
    builder.add_node("web", web_node)
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "rag")
    builder.add_edge(START, "web")
    builder.add_edge("rag", "answer")
    builder.add_edge("web", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def main():
    graph = build_graph()
    question = {"question": "LangGraph 是什么？"}

    #1. values：每一超步的完整状态（调试"状态对不对"最直观）
    print("① values —— 每个超步的完整状态")
    for i, state in enumerate(graph.stream(question, stream_mode="values")):
        print(f"   [{i}] keys={sorted(state.keys())}")

    #2. updates：每个节点返回的增量（只关心"谁改了什么"时用）
    print("\n② updates —— 每个节点返回的增量")
    for chunk in graph.stream(question, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"   {node}: {update}")

    #3. messages：模型 token 级流式（前端"打字机"效果）
    print("\n③ messages —— 模型输出逐 token")
    text = ""
    for message_chunk, metadata in graph.stream(question, stream_mode="messages"):
        if message_chunk.content:
            text += message_chunk.content
            print(f"   [{metadata.get('langgraph_node')}] {message_chunk.content!r}")
    print("   拼起来的回答：", text)

    #4. custom：节点里用 runtime.stream_writer 主动推的事件
    print("\n④ custom —— 业务自定义事件")
    for event in graph.stream(question, stream_mode="custom"):
        print("   ", event)

    #5. 多模式同时开：一次遍历拿到 updates + custom
    print("\n⑤ 组合模式 ['updates', 'custom']")
    for mode, payload in graph.stream(question, stream_mode=["updates", "custom"]):
        print(f"   [{mode}] {str(payload)[:70]}")


if __name__ == "__main__":
    main()
