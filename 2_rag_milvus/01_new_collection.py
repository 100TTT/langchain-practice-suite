from pymilvus import MilvusClient,DataType
# 定义一个方法创建collection
def create_collection(collection_name,schema,index_params):
    client = MilvusClient(
        uri="http://localhost:19530"
    )
    client.create_collection(
        collection_name = collection_name,
        schema=schema,
        index_params=index_params
     )
# 定义一个方法创建CollectionSchema
def create_collection_schema():
    schema = MilvusClient.create_schema(
        auto_id=True,
    ).add_field(
        field_name='id',
        datatype=DataType.INT64,
        is_primary=True,
    ).add_field(
        field_name='vector',
        datatype=DataType.FLOAT_VECTOR,
        dim=1024        # 稠密向量的维度大小
    ).add_field(
        field_name='sparse_vector',
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    ).add_field(
        field_name='text',
        datatype=DataType.VARCHAR,
        max_length=1500
    ).add_field(
        field_name='metadata',
        datatype=DataType.JSON
    )
    return schema
# 定义一个方法创建IndexParams(索引)  稠密与稀疏向量
def create_index_params():
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name='vector',
        index_type="HNSW",
        metric_type="L2"
    )
    index_params.add_index(
        field_name='sparse_vector',
        index_type='SPARSE_INVERTED_INDEX',
        metric_type='IP'
    )
    return index_params
if __name__ == "__main__":
    # 1.创建schema
    schema = create_collection_schema()
    # 2.创建index
    index_params = create_index_params()
    # 3.创建collection
    create_collection(
        collection_name = 'my_collection',
        schema=schema,
        index_params=index_params
    )


