"""10 · 人工介入：在敏感动作前把图停下来

考点：interrupt()、Command(resume=...)、checkpointer 与断点恢复、条件边分流
思路：转账这类动作不能全自动。审核节点先 interrupt 挂起整张图（返回 __interrupt__），
      人工决策后带 resume 再 invoke，图从断点继续，已完成的节点不重算
观察：同一张图，approve / reject / edit 三种决策走出三条路径，edit 还能改参数
运行：python 5_langgraph/10_interrupt_hitl.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt


class TransferState(TypedDict):
    amount: int
    to_account: str
    risk: str          # 风控结论
    audit: str         # approve / reject / edit
    status: str
    trace: Annotated[list, operator.add]


def risk_check(state: TransferState):
    # 真实项目里这里是一次风控模型/规则调用
    risk = "high" if state["amount"] > 500 else "low"
    return {"risk": risk, "trace": ["risk_check"]}


def human_audit(state: TransferState):
    # 关键一行：把图挂起，等外部把决策 resume 回来；返回值就是传进 Command(resume=) 的内容
    decision = interrupt(
        {
            "type": "approval_required",
            "amount": state["amount"],
            "to_account": state["to_account"],
            "risk": state["risk"],
            "hint": "approve 放行 / reject 拒绝 / edit 改参数后放行",
        }
    )

    if decision["type"] == "edit":
        edited = decision["edited_action"]
        return {
            "audit": "edit",
            "amount": edited["amount"],
            "to_account": edited["to_account"],
            "trace": ["human_audit:edit"],
        }
    return {"audit": decision["type"], "trace": [f"human_audit:{decision['type']}"]}


def execute_transfer(state: TransferState):
    print(f"    执行转账 {state['amount']} -> {state['to_account']}（风控={state['risk']}）")
    return {"status": "success", "trace": ["execute_transfer"]}


def cancel_transfer(state: TransferState):
    print(f"    已取消转账 {state['amount']} -> {state['to_account']}")
    return {"status": "cancelled", "trace": ["cancel_transfer"]}


def route_after_audit(state: TransferState):
    return "cancel_transfer" if state["audit"] == "reject" else "execute_transfer"


def build_graph():
    builder = StateGraph(state_schema=TransferState)
    builder.add_node("risk_check", risk_check)
    builder.add_node("human_audit", human_audit)
    builder.add_node("execute_transfer", execute_transfer)
    builder.add_node("cancel_transfer", cancel_transfer)

    builder.add_edge(START, "risk_check")
    builder.add_edge("risk_check", "human_audit")
    builder.add_conditional_edges("human_audit", route_after_audit)
    builder.add_edge("execute_transfer", END)
    builder.add_edge("cancel_transfer", END)

    # 断点续跑必须有 checkpointer：状态和"停在哪个节点"都要落库
    return builder.compile(checkpointer=InMemorySaver())


def run_scenario(graph, name, payload, decision):
    print(f"\n=== {name} ===")
    config = {"configurable": {"thread_id": name}}

    first = graph.invoke(payload, config)
    snapshot = graph.get_state(config)
    print(f"  第一次 invoke 返回 __interrupt__：{first.get('__interrupt__')}")
    print(f"  停下时 next={snapshot.next}（等 human_audit 恢复）")

    second = graph.invoke(Command(resume=decision), config)
    print(f"  恢复后 status={second['status']} audit={second['audit']} "
          f"amount={second['amount']}")
    print(f"  轨迹：{second['trace']}")


def main():
    graph = build_graph()

    run_scenario(graph, "approve-小额",
                 {"amount": 100, "to_account": "account_123"}, {"type": "approve"})

    run_scenario(graph, "reject-大额",
                 {"amount": 5000, "to_account": "account_999"}, {"type": "reject"})

    run_scenario(graph, "edit-改额度",
                 {"amount": 5000, "to_account": "account_999"},
                 {"type": "edit",
                  "edited_action": {"amount": 500, "to_account": "account_456"}})

    print(
        "\n要点："
        "\n  · interrupt 必须配 checkpointer，否则没有地方存「停在哪」"
        "\n  · resume 之前状态是持久的：可以隔几分钟、换一个进程再恢复"
        "\n  · 同 thread_id 才能续上；不同 thread 是另一笔请求，互不影响"
        "\n  · approve / reject / edit 是业务语义，interrupt 本身只负责「暂停 + 取值」"
    )


if __name__ == "__main__":
    main()
