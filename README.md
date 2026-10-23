# langchain-practice-suite

基于 LangChain + Milvus 的大模型应用实践仓库：覆盖模型调用与 LCEL 编排、文档解析与向量检索（RAG）、Agent 工具调用与 MCP 协议接入、会话记忆与人工介入中间件，以及 LangGraph 状态图编排（分支 / 并行 / 循环 / 人审 / 断点续跑）等模块，每个模块都可独立运行。

## 目录结构

```
langchain-practice-suite/
├── 1_core_lcel/                 # 模型调用、提示词模板、输出解析、LCEL 链与并行编排
│   ├── 01_load_env.py                # dotenv 加载环境变量
│   ├── 02_openai_start.py            # OpenAI SDK（Chat Completions）
│   ├── 03_openai_response_api.py     # OpenAI Responses API
│   ├── 04_langchain_start.py         # LangChain 统一入口
│   ├── 05_langchain_openai.py        # init_chat_model 调用在线模型
│   ├── 06_langchain_openai_async.py  # 异步调用 ainvoke
│   ├── 07_langchain_stream_call.py   # 流式输出 stream
│   ├── 08_langchain_batch_call.py    # 批量调用 batch
│   ├── 09_langchain_local_call.py    # Ollama 本地模型（ChatOllama）
│   ├── 10_langchain_response_parser.py   # StrOutputParser / JsonOutputParser
│   ├── 11_langchain_response_parser2.py  # PydanticOutputParser
│   ├── 12_langchain_prompt_template.py   # ChatPromptTemplate
│   ├── 13_chains.py                  # LCEL：模板 | 模型 | 解析器
│   ├── 14_langchain_runnable_paralle.py   # RunnableParallel
│   └── 14_langchain_runnable_paralle2.py  # 多模型并行 + 汇总（GPT / Gemini / DeepSeek）
├── 2_rag_milvus/                # 文档切分、嵌入、向量库、混合检索与重排
│   ├── 01_new_collection.py          # Schema：1024 维稠密 + 稀疏向量 + 元数据，HNSW/倒排索引
│   ├── 02_insert.py                  # 解析 → 切分 → BGE-M3 嵌入 → 入库
│   └── 03_search_data.py             # 稠密/稀疏双路召回 + RRF 重排 + 标量过滤
├── 3_agent_mcp/                 # Function Calling 三种写法 + MCP 服务端/客户端
│   ├── function_calling/
│   │   ├── call_tool_by_openai.py        # 原生 tools schema（strict）
│   │   ├── call_tool_by_langchain.py     # bind_tools 手工编排
│   │   ├── call_tool_by_create_agent.py  # create_agent 自动编排
│   │   └── demo.py                       # 结构化输出工具 + LangSmith 追踪
│   └── mcp/
│       ├── stdio_mcp_server.py / stdio_mcp_client.py     # 本地进程通信
│       ├── http_mcp_server.py / http_mcp_client.py       # Streamable HTTP
│       └── langchain_mcp_client.py / langchain_mcp_demo02.py  # LangChain 接入 MCP
├── 4_memory_middleware/         # 会话记忆与中间件
│   ├── short_term_memory/01_in_memory_saver_demo.py            # InMemorySaver 短期记忆
│   ├── long_term_memory/02_sqllite_saver_demo.py               # SqliteSaver 长期记忆
│   └── middleware/03_human_in_the_loop_middleware_demo.py      # 敏感工具人工审批
├── 5_langgraph/                 # 状态图编排：状态、Reducer、checkpointer、人审、缓存/重试/流式
│   ├── 01_state_graph_basics.py      # 七步建图 + input/output schema + 扇出扇入
│   ├── 02_private_state.py           # 节点私有状态（只声明自己用到的字段）
│   ├── 03_reducer.py                 # Annotated[list, add] / 自定义 reducer / InvalidUpdateError
│   ├── 04_checkpointer.py            # SqliteSaver、thread_id、get_state_history
│   ├── 05_superstep_parallel.py      # 超步模型：等深菱形 vs 深浅不一的扇入差异
│   ├── 06_runtime_context.py         # config 传参 + runtime.context 依赖注入
│   ├── 07_node_cache.py              # CachePolicy(ttl) 节点缓存
│   ├── 08_node_retry.py              # RetryPolicy 节点重试
│   ├── 09_stream_modes.py            # values / updates / messages / custom 四种流式
│   ├── 10_interrupt_hitl.py          # interrupt + Command(resume)：approve / reject / edit
│   ├── 11_loop_recursion.py          # 条件边成环与 recursion_limit 兜底
│   └── 12_capstone_workflow.py       # 综合工作流：分类 → 并行检索 → 自检回环 → 人审（离线可跑）
└── demo/                        # 演示脚手架（非课程练习代码，可自行替换或删除）
    └── app.py                   # Streamlit 交互式演示页
```

## 技术栈

- **框架**：LangChain（LCEL、Prompt Template、Output Parser、Agent）、LangGraph（StateGraph、Reducer、checkpointer、interrupt 人审、节点缓存 / 重试、stream）
- **模型**：OpenAI 兼容接口（GPT / Gemini / DeepSeek / Qwen）、Ollama 本地部署（qwen3:8b）
- **检索**：Milvus（HNSW + 稀疏倒排索引）、BGE-M3 稠密/稀疏双向量、RRF 融合重排
- **工具**：Function Calling、MCP（stdio / Streamable HTTP）、LangSmith 调用链追踪

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入自己的 API Key 与 Base URL
python 1_core_lcel/05_langchain_openai.py
python 5_langgraph/12_capstone_workflow.py   # LangGraph 综合工作流，离线可跑，不需要 API Key
```

- `5_langgraph/` 里的示例全部用 mock 模型，clone 下来直接能跑，方便先看清图结构；接真模型见该目录 README。

- RAG 模块需要本地 Milvus（standalone 模式，默认 `http://localhost:19530`），并把 BGE-M3 模型放到 `assets/models`；
- 本地模型模块需要先 `ollama serve` 并 `ollama pull qwen3:8b`；
- MCP 模块先启动 server 再运行 client。

## 在线演示（Streamlit）

`demo/app.py` 是一个可交互的 RAG 演示页（**演示脚手架，与上方的练习代码分开**）：上传文档 → 构建索引 → 提问 → 查看检索片段与带引用的回答，检索模式可在「混合检索 + RRF / 稠密向量 / 稀疏向量」之间切换。

```bash
pip install -r requirements.txt
streamlit run demo/app.py
# 浏览器打开 http://localhost:8501
```

运行前需要：Milvus 已启动、BGE-M3 模型已下载到 `assets/models/bge-m3`、`.env` 里配好 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`。

<!-- 录屏后把 GIF 放到 docs/demo.gif，再删掉下面这行的注释即可在页面顶部展示 -->
<!-- ![demo](docs/demo.gif) -->

## 说明

本项目为 LangChain 学习实践仓库：按模块逐个实现并跑通，其中 RAG 部分在示例基础上改写了切分策略与元数据设计（支持按章节做标量过滤），Agent 部分补充了 MCP 两种传输方式的对比实现，记忆部分对比了内存与 SQLite 两种 checkpointer 的差异。

`5_langgraph/` 是从「一条链」走到「状态图」的补充：LCEL 解决单链编排，LangGraph 解决分支、并行、循环与中断恢复。12 个示例逐个拆解单一能力，`12_capstone_workflow.py` 再把它们组装成一条完整工作流（分类路由 → 双路并行检索 → 生成自检回环 → 敏感动作人工审核 → 会话记忆），全部离线可跑。实现过程中踩到的坑（并行写同一 key 必须配 reducer、扇入在分支深浅不一时会被触发多次、interrupt 必须配 checkpointer、recursion_limit 数的是超步）都记录在该目录 README 里。

`demo/` 目录下是为了方便演示额外编写的脚手架页面，不属于课程练习代码，可自行替换或删除。
