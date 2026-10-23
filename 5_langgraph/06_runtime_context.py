"""06 · 运行时依赖：config 传业务参数，context 注入外部对象

考点：RunnableConfig（可配置项）、Runtime.context（依赖注入）、节点三分签名
思路：节点不该自己去 new 数据库和模型——把对象从 context 注入，把用户 id 从 config 传入，
      节点只负责取用；这样同一张图能跑不同用户、不同环境
观察：同一张图，config 换 user_id、context 换实现，就走不同分支
运行：python 5_langgraph/06_runtime_context.py
"""

from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime


#1. 状态：只放"会随执行变化"的数据
class SupportState(TypedDict):
    query: str
    response: str


#2. 外部依赖（真实项目里是数据库连接、向量库、模型客户端）
class MockLLM:
    def invoke(self, prompt: str) -> str:
        return f"(MockLLM) 关于「{prompt}」的详细说明……"


class MockUserDB:
    def get_user(self, user_id: str) -> dict:
        return {"id": user_id, "role": "vip" if "vip" in user_id else "standard"}


#3. 节点三分签名：state（数据）+ config（本次调用的配置）+ runtime（本次执行的依赖）
def customer_service_node(
    state: SupportState, config: RunnableConfig, runtime: Runtime
) -> dict:
    user_id = config["configurable"]["user_id"]
    db: MockUserDB = runtime.context["user_db"]
    llm: MockLLM = runtime.context["llm"]

    user = db.get_user(user_id)
    if user["role"] == "vip":
        # 会员走详细回答
        return {"response": f"[VIP] {llm.invoke(state['query'])}"}
    # 普通用户走模板回复
    return {"response": f"[普通] 已收到「{state['query']}」，稍后回复您"}


def build_graph():
    builder = StateGraph(state_schema=SupportState)
    builder.add_node("customer_service", customer_service_node)
    builder.add_edge(START, "customer_service")
    builder.add_edge("customer_service", END)
    return builder.compile()


def main():
    graph = build_graph()

    #4. 同一张图，不同用户走不同分支
    for user_id in ("user_vip_123", "user_456"):
        config = {"configurable": {"user_id": user_id}}
        context = {"llm": MockLLM(), "user_db": MockUserDB()}
        result = graph.invoke(
            {"query": "我想了解一下会员权益"}, config=config, context=context
        )
        print(f"{user_id:12s} -> {result['response']}")

    print(
        "\n要点："
        "\n  · config 是「本次调用」的可配置项，会参与缓存键，适合 thread_id / user_id"
        "\n  · context 是「本次执行」注入的依赖对象，不参与状态、也不落检查点"
        "\n  · 依赖注入让节点可测：单测时换一个 MockLLM 就行，不用打桩全局变量"
    )


if __name__ == "__main__":
    main()
