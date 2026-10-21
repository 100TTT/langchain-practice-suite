#1. 定义一个方法将用户的问题(字符串)转成向量(稠密和稀疏)
from typing import List

from FlagEmbedding import BGEM3FlagModel
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker


def question_to_vector(question):
    #1.1 创建模型对象
    client = BGEM3FlagModel(model_name_or_path="../assets/models/bge-m3")
    #1.2 将问题转成向量
    all_vectors = client.encode(question,return_dense=True,return_sparse=True)
    return all_vectors

#定义一个创建MilvusClient对象的方法
def create_milvus_client() -> MilvusClient:
    return MilvusClient(uri="http://localhost:19530")

#2. 定义一个根据稠密向量进行检索的方法
def search_by_dense_vector(client:MilvusClient,dense_vector,top_k=5) -> List[List[dict]]:
    results = client.search(
        collection_name="my_collection",
        limit=top_k,
        anns_field="vector",
        data=[dense_vector],
        search_params={
            "metric_type": "L2"
        },
        output_fields=["text","metadata"]
    )
    print(results)
    return results

#3. 定义一个根据稀疏向量进行检索的方法
def search_by_sparse_vector(client:MilvusClient,sparse_vector,top_k=5):
    results = client.search(
        collection_name="my_collection",
        limit=top_k,
        anns_field="sparse_vector",
        data=[sparse_vector],
        search_params={
            "metric_type": "IP"
        },
        output_fields=["text", "metadata"]
    )
    return results

#4. 定义一个进行混合检索的方法
def hybrid_search(client:MilvusClient,dense_vector,sparse_vector,top_k=5):
    #4.1 定义根据稠密向量检索的请求和根据稀疏向量检索的请求
    dense_search_request = AnnSearchRequest(
        data=[dense_vector],
        anns_field="vector",
        limit=top_k,
        param={
            "metric_type": "L2"
        }
    )

    sparse_search_request = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        limit=top_k,
        param={
            "metric_type": "IP"
        }
    )
    #4.2 发起混合检索
    results = client.hybrid_search(
        collection_name="my_collection",
        reqs=[dense_search_request,sparse_search_request],
        output_fields=["text", "metadata"],
        limit=top_k,
        ranker=RRFRanker()
    )
    return results

#5. 定义一个主方法，将用户问题转成向量后进行检索，最后将检索内容传给大模型
def main(question:str):
    #5.1 将问题转成向量
    all_vectors = question_to_vector(question)

    # #5.2 根据稠密向量进行检索
    # client = create_milvus_client()
    # results = search_by_dense_vector(client,all_vectors['dense_vecs'],5)

    # 5.2 根据稀疏向量进行检索
    # client = create_milvus_client()
    # results = search_by_sparse_vector(client,all_vectors['lexical_weights'],5)

    #5.2 根据稠密向量和稀疏向量进行混合检索
    client = create_milvus_client()
    results = hybrid_search(client,all_vectors['dense_vecs'],all_vectors['lexical_weights'],5)

    # 5.3 将检索结果和用户的问题进行拼接
    context = "\n".join([res['entity']['text'] for res in results[0]])

    #5.4 传给大模型进行回答
    llm = ChatOpenAI(model="gpt-4o-mini")
    res = llm.invoke(
        [
            {"role": "system",
             "content": "你是一个专业的法律问答机器人，请根据上下文回答问题，当上下文无法回答问题时，请回答“根据上下文无法回答该问题”"},
            {"role": "user", "content": f"根据以下上下文回答问题：{context}\n问题：{question}"}
       ]
    )

    print(res)

if __name__ == "__main__":
    main("监护人有哪些职责？")