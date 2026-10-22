# -*- coding: utf-8 -*-
"""
Streamlit 演示页：知识库 RAG 问答（Milvus + BGE-M3 + 大模型）

注意：本文件是方便演示而编写的脚手架页面，不属于课程练习代码。

运行：
    pip install -r requirements.txt
    streamlit run demo/app.py

依赖服务：
    - Milvus：默认 http://localhost:19530（standalone 模式）
    - BGE-M3 模型：默认 ./assets/models/bge-m3
    - 大模型：在 .env 里配置 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
"""
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "demo_docs")
DEFAULT_MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DEFAULT_MODEL_DIR = os.getenv("BGE_M3_DIR", "./assets/models/bge-m3")
DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DIM = 1024

st.set_page_config(page_title="知识库 RAG 问答 Demo", page_icon="📚", layout="wide")


# --------------------------------------------------------------------------
# 资源加载（带缓存）
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="正在加载嵌入模型 BGE-M3 ...")
def load_embedder(model_dir):
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(model_name_or_path=model_dir, use_fp16=True)


@st.cache_resource(show_spinner="正在连接 Milvus ...")
def load_milvus(uri):
    from pymilvus import MilvusClient

    return MilvusClient(uri=uri)


@st.cache_resource(show_spinner="正在初始化大模型 ...")
def load_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.2,
    )


# --------------------------------------------------------------------------
# 文档处理
# --------------------------------------------------------------------------
def read_file(uploaded):
    """把上传的文件读成纯文本。"""
    name = uploaded.name.lower()
    suffix = os.path.splitext(name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.getbuffer())
        path = f.name

    try:
        if suffix in (".txt", ".md"):
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        if suffix == ".docx":
            import docx

            return "\n".join(p.text for p in docx.Document(path).paragraphs)
        if suffix == ".pdf":
            from pypdf import PdfReader

            return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
        return ""
    finally:
        os.unlink(path)


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按段落/句号递归切分，保证语义相对完整。"""
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end]
        # 尽量在句号、换行处断开
        for mark in ("。", "\n", "；", "！", "？"):
            idx = piece.rfind(mark)
            if idx > size * 0.5:
                piece = piece[: idx + 1]
                end = start + len(piece)
                break
        chunks.append(piece.strip())
        start = max(end - overlap, end) if end == start + size else end
        start = end - overlap if end - overlap > start else end
    return [c for c in chunks if c]


def ensure_collection(client, name):
    from pymilvus import DataType, MilvusClient

    if client.has_collection(name):
        return
    schema = (
        MilvusClient.create_schema(auto_id=True)
        .add_field("id", DataType.INT64, is_primary=True)
        .add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
        .add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        .add_field("text", DataType.VARCHAR, max_length=2000)
        .add_field("metadata", DataType.JSON)
    )
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="HNSW", metric_type="L2",
                           params={"M": 16, "efConstruction": 200})
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)


def embed(embedder, texts, is_query=False):
    out = embedder.encode(texts, return_dense=True, return_sparse=True,
                          max_length=512 if is_query else 1024)
    return out["dense_vecs"], out["lexical_weights"]


def insert_docs(client, embedder, collection, name, text):
    chunks = split_text(text)
    dense, sparse = embed(embedder, chunks)
    rows = [
        {"vector": list(map(float, d)), "sparse_vector": {str(k): float(v) for k, v in s.items()},
         "text": c[:2000], "metadata": {"source": name, "chunk": i}}
        for i, (c, d, s) in enumerate(zip(chunks, dense, sparse))
    ]
    client.insert(collection_name=collection, data=rows)
    return len(rows)


def search(client, embedder, collection, question, top_k, mode):
    from pymilvus import AnnSearchRequest, RRFRanker

    dense_vec, sparse_vec = embed(embedder, [question], is_query=True)
    dense_vec, sparse_vec = list(map(float, dense_vec[0])), sparse_vec[0]

    if mode == "稠密向量":
        res = client.search(collection_name=collection, data=[dense_vec], anns_field="vector",
                            limit=top_k, search_params={"metric_type": "L2"}, output_fields=["text", "metadata"])
        return [[(hit["entity"]["text"], hit["distance"], hit["entity"]["metadata"]) for hit in hits] for hits in res][0]

    if mode == "稀疏向量":
        res = client.search(collection_name=collection, data=[{str(k): float(v) for k, v in sparse_vec.items()}],
                            anns_field="sparse_vector", limit=top_k,
                            search_params={"metric_type": "IP"}, output_fields=["text", "metadata"])
        return [[(hit["entity"]["text"], hit["distance"], hit["entity"]["metadata"]) for hit in hits] for hits in res][0]

    # 混合检索 + RRF 重排
    reqs = [
        AnnSearchRequest(data=[dense_vec], anns_field="vector", param={"metric_type": "L2"}, limit=top_k * 2),
        AnnSearchRequest(data=[{str(k): float(v) for k, v in sparse_vec.items()}], anns_field="sparse_vector",
                         param={"metric_type": "IP"}, limit=top_k * 2),
    ]
    res = client.hybrid_search(collection_name=collection, reqs=reqs, ranker=RRFRanker(60),
                               limit=top_k, output_fields=["text", "metadata"])
    return [[(hit["entity"]["text"], hit["distance"], hit["entity"]["metadata"]) for hit in hits] for hits in res][0]


def answer(llm, question, hits):
    context = "\n\n".join(f"[{i + 1}] {h[0]}" for i, h in enumerate(hits))
    prompt = (
        "你是知识库问答助手。只能依据下面的资料回答，不要编造；"
        "回答时用 [编号] 标注依据，找不到答案就说「资料中没有相关内容」。\n\n"
        f"资料：\n{context}\n\n问题：{question}"
    )
    return llm.invoke(prompt).content


# --------------------------------------------------------------------------
# 页面
# --------------------------------------------------------------------------
st.title("📚 知识库 RAG 问答 Demo")
st.caption("Milvus 双路召回（稠密 + 稀疏）· RRF 重排 · 大模型生成与引用")

with st.sidebar:
    st.subheader("⚙️ 配置")
    milvus_uri = st.text_input("Milvus 地址", DEFAULT_MILVUS_URI)
    collection = st.text_input("Collection", DEFAULT_COLLECTION)
    model_dir = st.text_input("BGE-M3 模型目录", DEFAULT_MODEL_DIR)
    top_k = st.slider("检索条数 Top-K", 1, 10, 4)
    mode = st.radio("检索模式", ["混合检索 + RRF", "稠密向量", "稀疏向量"], index=0)
    st.divider()
    st.caption(f"大模型：{DEFAULT_LLM_MODEL}")

tab_build, tab_ask = st.tabs(["① 构建知识库", "② 提问"])

with tab_build:
    files = st.file_uploader("上传文档（txt / md / docx / pdf）", type=["txt", "md", "docx", "pdf"],
                             accept_multiple_files=True)
    if st.button("构建索引", type="primary", disabled=not files):
        try:
            embedder, client = load_embedder(model_dir), load_milvus(milvus_uri)
            ensure_collection(client, collection)
            total = 0
            for f in files:
                total += insert_docs(client, embedder, collection, f.name, read_file(f))
            st.success(f"已写入 {total} 个片段到 `{collection}`")
        except Exception as e:  # noqa: BLE001
            st.error(f"构建失败：{type(e).__name__}: {e}")
            st.info("请确认 Milvus 已启动（默认 19530），且 BGE-M3 模型目录存在。")

with tab_ask:
    question = st.text_input("输入问题", placeholder="例如：四等水准测量的闭合差限差是多少？")
    if st.button("提问", type="primary", disabled=not question):
        try:
            embedder, client = load_embedder(model_dir), load_milvus(milvus_uri)
            hits = search(client, embedder, collection, question, top_k, mode)
            if not hits:
                st.warning("没有检索到内容，请先构建知识库。")
            else:
                with st.spinner("生成回答中 ..."):
                    st.markdown("### 回答")
                    st.write(answer(load_llm(), question, hits))
                st.markdown("### 检索片段")
                for i, (text, score, meta) in enumerate(hits, 1):
                    with st.expander(f"[{i}] 相似度 {score:.4f} · 来源 {meta.get('source', '-')}"):
                        st.write(text)
        except Exception as e:  # noqa: BLE001
            st.error(f"检索失败：{type(e).__name__}: {e}")
