from pymilvus import MilvusClient
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel

# 定义一个插入数据的方法
def insert_data():
    # 1.加载文件
    loader = Docx2txtLoader(
        file_path='./assets/sample.docx'
        )
    docs = loader.load()
    # 2.切分文件
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,  
        separators = ['\n\n','\n','.']
    )
    splitted_docs = splitter.split_documents(docs)[0:20]  # 返回list[document]
    # 3.转成向量
    # 3.1创建模型对象
    model = BGEM3FlagModel(
        model_name_or_path ='./assets/models/bge-m3'
    )
    # 3.2 转成向量,稠密和稀疏向量都要
    all_vectors = model.encode([doc.page_content for doc in splitted_docs],return_dense=True,return_sparse=True)   # encode中需要str类型 list[str] 遍历取即为str  doc.是Langchain的内置属性
    dense_vecs = all_vectors['dense_vecs']  # 多个文档的稠密向量
    lexical_weights = all_vectors['lexical_weights'] # 多个文档的稀疏向量
    datas = []
    # 3.3 构建要插入数据库的数据 
    for doc,dense_vec,sparse_vec in zip(splitted_docs,dense_vecs,lexical_weights): 
        datas.append( 
            {
                'vector':dense_vec,
                'sparse_vector':sparse_vec,
                'text':doc.page_content,
                'metadata':doc.metadata   
            }
        )
    # 4.插入数据库
    # 4.1 创建MilvusClient对象
    client = MilvusClient(
        url='http://localhost:19530',
    )
    client.insert(
        collection_name = 'my_collection',
        data = datas
    )
    # 5.
if __name__ == '__main__':
    insert_data()
