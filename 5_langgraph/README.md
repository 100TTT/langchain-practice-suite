# 5_langgraph —— 用状态图编排多步工作流

LCEL 是「一条链走到底」。一旦出现下面这些需求，链就写不下去了：

- 按问题类型分流（知识问答 / 工具调用 / 闲聊）
- 两路检索并行、结果一起汇总
- 回答不合格要回炉重写
- 退款、写库这类敏感动作必须停下来等人确认，确认完从断点继续

LangGraph 的思路是把工作流写成**状态图**：节点是函数、只负责改状态；边决定下一步去哪。于是分支、循环、中断、恢复都变成图结构问题，而不是散落在代码里的 if/while/try。

本模块把 12 个最小示例按能力拆开，最后用一个综合工作流把它们串起来。

## 目录

| 文件 | 考点 | 关键 API |
|---|---|---|
| `01_state_graph_basics.py` | 七步建图：状态 → 节点 → 加边 → 编译 → invoke；扇出与扇入 | `StateGraph`、`START`、`END`、`input_schema` / `output_schema` |
| `02_private_state.py` | 节点只声明自己用到的字段 | 节点级 `TypedDict` |
| `03_reducer.py` | 并行写同一 key 怎么合并；不配 reducer 会怎样 | `Annotated[list, operator.add]`、自定义 reducer、`InvalidUpdateError` |
| `04_checkpointer.py` | 会话记忆、thread 隔离、历史回放 | `SqliteSaver` / `InMemorySaver`、`thread_id`、`get_state`、`get_state_history` |
| `05_superstep_parallel.py` | 超步执行模型；扇入何时只跑一次 | `state.channels`（等深菱形 vs 深浅不一的对比） |
| `06_runtime_context.py` | 配置与依赖注入、节点三分签名 | `RunnableConfig`、`Runtime`（`runtime.context`） |
| `07_node_cache.py` | 给慢节点加 TTL 缓存 | `CachePolicy(ttl=...)`、`InMemoryCache` |
| `08_node_retry.py` | 节点级重试 | `RetryPolicy(max_attempts=...)` |
| `09_stream_modes.py` | 四种流式模式各给谁用 | `stream_mode=values/updates/messages/custom`、`runtime.stream_writer` |
| `10_interrupt_hitl.py` | 人工审核与断点续跑 | `interrupt()`、`Command(resume=...)` |
| `11_loop_recursion.py` | 环、终止条件、死循环兜底 | `add_conditional_edges` 成环、`recursion_limit`、`GraphRecursionError` |
| `12_capstone_workflow.py` | 把上面全部组装成一条工作流 | 以上全部 |

## 运行

```bash
pip install -r requirements.txt      # 本模块只依赖 langgraph / langchain-core
python 5_langgraph/12_capstone_workflow.py
```

- `01`–`12` **全部离线可跑**，不需要 API Key：模型调用处用固定返回值或 `FakeListChatModel` 代替，方便先看清图的结构。
- 想接真模型，把 `09`、`12` 里的假模型换成 `init_chat_model("openai:gpt-4o-mini")`（读 `.env` 里的 `OPENAI_API_KEY` / `OPENAI_BASE_URL`）即可，图结构不用改。
- `04_checkpointer.py` 会在同目录生成 `checkpoint.db`（已在 `.gitignore` 里忽略）。

## 概念速查

- **状态（State）**：一个 `TypedDict`，就是整张图共享的黑板。节点返回的 dict 只会合并进对应字段。
- **超步（superstep）**：一波节点并行跑完 → 更新合并进状态 → 下一波。`recursion_limit` 数的就是这个「波数」。
- **通道（channel）**：状态里每个字段对应一个通道。默认通道「一步只收一个值」，所以要并行写就必须用 reducer 声明合并规则。
- **checkpointer**：把每个超步的状态落库，`thread_id` 决定会话。记忆、断点续跑、时间旅行都靠它。
- **interrupt / resume**：节点里调用 `interrupt()` 会让整张图挂起，人工决策后用 `Command(resume=...)` 从断点继续，已完成的节点不重算。
- **节点级策略**：缓存、重试、超时都挂在 `add_node(...)` 上，只影响那一个节点，不是全局开关。

## 踩坑记录（本模块实测）

1. **并行写同一个 key 不配 reducer，直接报错**
   ```
   InvalidUpdateError: At key 'aggregate': Can receive only one value per step.
   Use an Annotated key to handle multiple values.
   ```
   见 `03_reducer.py` 的第 ③ 种写法。reducer 是合并协议，不是可选装饰。

2. **扇入 ≠ 「等所有分支都完成」**
   等深菱形（两条支路步数相同）时，汇聚节点只执行 1 次；分支深浅不一时，先到的那条会先把汇聚节点触发一遍，等另一条到了再触发一遍——`05_superstep_parallel.py` 里 `d` 的日志会打印两次，`aggregate` 里出现两个 `d`。
   实践影响：汇聚节点要么幂等，要么把分支压成等深，或者用 reducer 去重。

3. **`interrupt()` 必须配 checkpointer**
   否则没有地方记录「停在哪」，`Command(resume=...)` 无从恢复。见 `10_interrupt_hitl.py`。

4. **`recursion_limit` 数的是超步，不是节点**
   一轮 `draft → improve` 循环占 2 步，设 8 只能跑 4 轮。见 `11_loop_recursion.py`。

5. **缓存只需要 `cache=`，不需要 checkpointer**
   `04` 的 checkpointer 管「会话记忆」，`07` 的 cache 管「节点结果复用」，两件事别混。缓存键 ≈ 节点名 + 该节点看到的输入，所以换输入、改节点名都会失效；有副作用的节点不要加缓存。

6. **`output_schema` 会裁剪返回值**
   调 `invoke` 只会拿到 `output_schema` 声明的字段，中间状态留在图里，不会泄漏给调用方。想调试就用 `get_state()` 或 `stream_mode="values"`。

7. **`config` 和 `context` 不是一回事**
   `config` 是本次调用的可配置项（`thread_id`、`user_id`），会参与缓存键；`context` 是本次执行注入的依赖对象（模型、数据库），不进状态、不落检查点。

## 综合工作流（12 号文件）长什么样

```
                       ┌─ rag_search ─┐
 START → classify ─────┤              ├→ generate → review ─pass──────→ END
            │          └─ web_search ─┘      ↑          │
            │                                  └─rewrite─┘
            ├→ tool_lookup → generate … → approve_action(interrupt) → execute / cancel → END
            └→ direct_reply → END
```

演示里能看到的四件事：

1. **分类路由**：`classify` 用规则判断走检索、工具还是闲聊；条件边返回列表就能扇出到两个检索节点。
2. **自检回环**：`generate` 第一稿故意不写来源，`review` 判不合格 → 回到 `generate` 重写一次 → 再判通过（`rewrites` 记次数，防止无限重写）。
3. **人工审核**：工具类请求生成退款单后停在 `approve_action`，`Command(resume={"type": "approve"})` 之后才执行。
4. **会话记忆**：同一个 `thread_id` 连问两轮，回答里的「第 N 轮」会递增。

## 和 `4_memory_middleware/` 的关系

`4_memory_middleware` 是框架封装路线：`create_agent` + `HumanInTheLoopMiddleware`，几行就能给 Agent 加记忆和审批，适合「标准 Agent + 少量定制」。

本模块是原生路线：自己用 `StateGraph` 定义状态、边和中断点，适合「流程本身就有业务形状」（分流、并行、回环、多种终止条件）。两条路线都能上生产，区别在于你把控制权交给框架多少。

## 说明

示例是在学习尚硅谷 LangGraph 课程与官方文档后重写的：保留了课程里的主线例子（七步建图、Reducer、checkpointer、interrupt、循环），补上了课程外自己实测出来的结论（扇入触发次数、`InvalidUpdateError` 原文、缓存与 checkpointer 的分工），并把所有外部依赖替换成可复现的 mock，保证 clone 下来就能跑。

依赖版本：`langgraph 1.2.x`、`langchain 1.4.x`（见根目录 `requirements.txt`）。
