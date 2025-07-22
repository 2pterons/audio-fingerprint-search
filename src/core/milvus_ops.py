from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
import argparse

def init_collection(model_name, user_name, dim=768):
    collection_name = f"{user_name}_audio_segments"
    fields = [
        FieldSchema(name="segment_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="audio_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=10),
    ]
    schema = CollectionSchema(fields, description=f"{model_name} audio embeddings")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }

    collection.create_index("embedding", index_params)
    collection.load()
    print(f"Milvus 컬렉션 생성 완료: {collection_name}")
