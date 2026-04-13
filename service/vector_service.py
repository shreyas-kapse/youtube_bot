import logging

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from db.qdrant_client import QdrantSingleton
from service.embedding_service import EmbeddingService

try:
    from enums import CollectionName
except ImportError:
    from ..enums import CollectionName
from qdrant_client.models import FieldCondition, Filter, MatchValue, VectorParams, Distance

logger = logging.getLogger(__name__)
class VectorService:

    def __init__(self):
        client = QdrantSingleton().client
        embedding_model = EmbeddingService.get_model()
        self.create_collection(client)
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=CollectionName.YouTube_Video_Transcript.value,
            embedding=embedding_model
        )
        
    def create_collection(self, client: QdrantClient):
        collections = client.get_collections().collections
        names = [c.name for c in collections]
        if CollectionName.YouTube_Video_Transcript.value not in names:
            client.recreate_collection(
                collection_name=CollectionName.YouTube_Video_Transcript.value,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )
            logger.info("Qdrant collection created")

    def add_documents(self, docs):
        if not docs:
            return
        video_id = docs[0].metadata.get("video_id")

        existing = self.vector_store.client.count(
            collection_name=self.vector_store.collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.video_id",
                        match=MatchValue(value=video_id)
                    )
                ]
            )
        )

        if existing.count > 0:
            logger.info(f"Video {video_id} already embedded. Skipping...")
            return

        self.vector_store.add_documents(docs)

    def get_retriever(self, video_id: str, k=10):
        return self.vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {
                    "must": [
                        {
                            "key": "metadata.video_id",
                            "match": {"value": video_id}
                        }
                    ]
                }
            }
        )
        
    def search(self, query: str, video_id: str, k: int = 10):
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter={
                "must": [
                    {
                        "key": "metadata.video_id",
                        "match": {"value": video_id}
                    }
                ]
            }
        )

        return results