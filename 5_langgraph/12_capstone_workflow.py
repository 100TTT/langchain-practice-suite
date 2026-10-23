"""12 · 综合工作流：分类路由 + 双路并行检索 + 生成自检回环 + 人工审核 + 会话记忆

把 01-11 的单项能力组装成一条能演示的完整工作流（离线可跑，不需要 API Key）：

                          ┌─ rag_search ─┐
  START → classify ───────┤              ├→ generate → review ─pass──────────┐
             │            └─ web_search ─┘       ↑          │                │
             │                                    └─rewrite─┘                │
             ├→ tool_lookup → generate … → approve_action(interrupt) → execute / cancel
             └→ direct_reply → END

用到的能力：
  01 状态与 input/output schema      02 节点私有状态        03 reducer（trace / history 追加）
  04 checkpointer + thread_id        05 并行扇出与同超步扇入  07 CachePolicy（web 检索缓存）
  08 RetryPolicy（generate 重试）     09 stream_mode="custom" 10 interrupt + Command(resume)
  11 条件边成环（生成 → 自检 → 重写）

运行：python 5_langgraph/12_capstone_workflow.py
"""

import operator
import time
from typing import Annotated, Literal, TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.types import CachePolicy, Command, RetryPolicy, interrupt


# ---------------------------------------------------------------- 状态
class WorkflowState(TypedDict):
    question: str
    route: str                 # knowledge / tool / chat
    rag_result: str
    web_result: str
    tool_result: str
    answer: str
    verdict: str               # pass / rewrite
    rewrites: int              # 已经重写过几次
    action: str                # 需要人工审核的敏感动作
    audit: str                 # approve / reject
    status: str
    history: Annotated[list, operator.add]   # 会话记忆（thread 内累积）
    trace: Annotated[list, operator.add]     # 执行轨迹（并行分支会同时追加）


# ---------------------------------------------------------------- 模拟外部系统
KB = {
    "条件边": "条件边用 add_conditional_edges 注册路由函数，函数返回下一个节点名；返回列表可以扇出到多个节点。",
    "循环": "图里的环靠条件边形成，必须写显式终止条件，并用 recursion_limit 兜底。",
    "检查点": "checkpointer 把每个超步的状态落库，thread_id 隔离会话，因此可以断点续跑。",
}

ORDER_DB = {"A1001": {"status": "已签收", "amount": 88, "refundable": True}}


def search_kb(question: str) -> str:
    hits = [f"[{i + 1}] {doc}" for i, (key, doc) in enumerate(KB.items()) if key in question]
    return " ".join(hits) if hits else "[1] 知识库里没有直接匹配的条目"


# ---------------------------------------------------------------- 节点
def classify(state: WorkflowState):
    """规则分类（真实项目可换成模型分类）。"""
    q = state["question"]
    if any(k in q for k in ("退款", "订单", "改地址", "取消")):
        route = "tool"
    elif any(k in q for k in ("你好", "在吗", "谢谢")):
        route = "chat"
    else:
        route = "knowledge"
    print(f"  · classify → {route}")
    # action 每轮清空，避免上一轮的敏感动作污染本轮
    return {"route": route, "action": "", "history": [q], "trace": [f"classify:{route}"]}


def rag_search(state: WorkflowState, runtime: Runtime):
    runtime.stream_writer({"stage": "retrieve", "detail": "检索知识库"})
    return {"rag_result": search_kb(state["question"]), "trace": ["rag_search"]}


def web_search(state: WorkflowState, runtime: Runtime):
    runtime.stream_writer({"stage": "retrieve", "detail": "检索网络"})
    time.sleep(0.2)  # 模拟一次慢查询（配合 CachePolicy 演示命中）
    return {"web_result": f"[Web] 「{state['question']}」的社区实践 3 条", "trace": ["web_search"]}


def tool_lookup(state: WorkflowState):
    """查询订单，为敏感动作准备参数。"""
    order_id = next((k for k in ORDER_DB if k in state["question"]), None)
    if not order_id:
        return {"tool_result": "未找到相关订单", "trace": ["tool_lookup:miss"]}
    order = ORDER_DB[order_id]
    return {
        "tool_result": f"订单 {order_id}：{order['status']}，金额 {order['amount']} 元",
        "action": f"为订单 {order_id} 发起退款 {order['amount']} 元",
        "trace": [f"tool_lookup:{order_id}"],
    }


def generate(state: WorkflowState):
    """生成回答；知识类第一次故意漏引用，用来演示自检回环。"""
    turn = len(state["history"])
    if state["route"] == "tool":
        answer = f"（第 {turn} 轮）{state['tool_result']}，已为你准备退款单，等待人工审核。"
    else:
        body = f"（第 {turn} 轮）{state['rag_result']}｜{state['web_result']}"
        # 第一次生成不写来源，让 review 判为不合格 → 触发一次重写
        answer = body if state["rewrites"] == 0 else body + "\n来源：" + state["rag_result"]
    print(f"  · generate（第 {state['rewrites'] + 1} 稿，{turn} 轮对话）")
    return {"answer": answer, "trace": [f"generate#{state['rewrites'] + 1}"]}


def review(state: WorkflowState):
    """自检：知识类必须带来源，最多重写 1 次。"""
    if state["route"] == "tool":
        return {"verdict": "pass", "trace": ["review:skip(tool)"]}
    if "来源：" in state["answer"]:
        return {"verdict": "pass", "trace": ["review:pass"]}
    if state["rewrites"] >= 1:
        print("  · review 重写次数已达上限，放行")
        return {"verdict": "pass", "trace": ["review:pass(上限)"]}
    print("  · review 判定缺少来源 → 重写")
    return {"verdict": "rewrite", "rewrites": state["rewrites"] + 1, "trace": ["review:rewrite"]}


def approve_action(state: WorkflowState):
    """敏感动作：挂起整张图，等人工放行。"""
    decision = interrupt({"type": "approval_required", "action": state["action"],
                          "detail": state["tool_result"]})
    return {"audit": decision["type"], "trace": [f"approve:{decision['type']}"]}


def execute_action(state: WorkflowState):
    print(f"  · 执行：{state['action']}")
    return {"status": "executed", "trace": ["execute_action"]}


def cancel_action(state: WorkflowState):
    print(f"  · 取消：{state['action']}")
    return {"status": "cancelled", "trace": ["cancel_action"]}


def direct_reply(state: WorkflowState):
    return {"answer": "你好，我是 LangGraph 实践仓库里的演示助手。", "trace": ["direct_reply"]}


# ---------------------------------------------------------------- 路由
def route_after_classify(state: WorkflowState):
    if state["route"] == "knowledge":
        return ["rag_search", "web_search"]   # 返回列表 = 扇出到两个节点
    return "tool_lookup" if state["route"] == "tool" else "direct_reply"


def route_after_review(state: WorkflowState) -> str:
    if state["verdict"] == "rewrite":
        return "generate"                      # 成环：回到生成节点
    if state["action"]:
        return "approve_action"                # 有敏感动作就找人审
    return END


def route_after_audit(state: WorkflowState) -> Literal["execute_action", "cancel_action"]:
    return "execute_action" if state["audit"] == "approve" else "cancel_action"


# ---------------------------------------------------------------- 构图
def build_graph():
    builder = StateGraph(state_schema=WorkflowState)

    builder.add_node("classify", classify)
    builder.add_node("rag_search", rag_search)
    # 慢且幂等 → 加 TTL 缓存
    builder.add_node("web_search", web_search, cache_policy=CachePolicy(ttl=60))
    builder.add_node("tool_lookup", tool_lookup)
    # 生产里这里是一次 LLM 调用 → 配重试
    builder.add_node("generate", generate, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("review", review)
    builder.add_node("approve_action", approve_action)
    builder.add_node("execute_action", execute_action)
    builder.add_node("cancel_action", cancel_action)
    builder.add_node("direct_reply", direct_reply)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify)

    builder.add_edge("rag_search", "generate")
    builder.add_edge("web_search", "generate")
    builder.add_edge("tool_lookup", "generate")

    builder.add_edge("generate", "review")
    builder.add_conditional_edges("review", route_after_review)
    builder.add_conditional_edges("approve_action", route_after_audit)

    builder.add_edge("execute_action", END)
    builder.add_edge("cancel_action", END)
    builder.add_edge("direct_reply", END)

    # 记忆 + 缓存：checkpointer 管会话，cache 管节点复用
    return builder.compile(checkpointer=InMemorySaver(), cache=InMemoryCache())


# ---------------------------------------------------------------- 演示
def show_stream(graph, payload, config):
    for mode, chunk in graph.stream(payload, config, stream_mode=["custom", "updates"]):
        if mode == "custom":
            print(f"    进度 ← {chunk['stage']}：{chunk['detail']}")
        else:
            for node, update in chunk.items():
                if node == "__interrupt__":
                    print(f"    挂起 ← {update[0].value['action']}")
                else:
                    print(f"    {node} → {list(update) if isinstance(update, dict) else update}")


def main():
    graph = build_graph()
    thread_kb = {"configurable": {"thread_id": "demo-kb"}}

    print("场景 1：知识类问题（并行检索 → 自检不合格 → 重写 → 通过）")
    show_stream(graph, {"question": "LangGraph 的条件边怎么用？", "rewrites": 0}, thread_kb)
    final = graph.get_state(thread_kb).values
    print("    回答：", final["answer"].replace("\n", " ⏎ "))
    print("    轨迹：", " → ".join(final["trace"]))

    print("\n场景 2：同 thread 追问（验证会话记忆：轮次应递增）")
    show_stream(graph, {"question": "那循环怎么保证不炸？", "rewrites": 0}, thread_kb)
    final2 = graph.get_state(thread_kb).values
    print("    回答：", final2["answer"].splitlines()[0])

    print("\n场景 3：工具类问题（触发人工审核，approve 后执行）")
    cfg = {"configurable": {"thread_id": "demo-tool"}}
    show_stream(graph, {"question": "帮我退款订单 A1001，金额 88 元", "rewrites": 0}, cfg)
    pending = graph.get_state(cfg)
    print("    待办节点：", pending.next, "（图停在这里等人）")
    resumed = graph.invoke(Command(resume={"type": "approve"}), cfg)
    print("    审核结果：", resumed["status"], "｜轨迹：", " → ".join(resumed["trace"]))

    print("\n场景 4：闲聊（直接回复，不进检索）")
    chat = graph.invoke({"question": "你好呀", "rewrites": 0},
                        {"configurable": {"thread_id": "demo-chat"}})
    print("    ", chat["answer"])

    print(
        "\n说明："
        "\n  · demo-kb 线程连问两次，回答里的「第 N 轮」递增 = checkpointer 在起作用"
        "\n  · demo-tool 线程停在了 approve_action，Command(resume) 之后才继续，已完成节点不重算"
        "\n  · web_search 在同一问题重复出现时会命中 TTL 缓存，可把 time.sleep 调大观察"
    )


if __name__ == "__main__":
    main()
