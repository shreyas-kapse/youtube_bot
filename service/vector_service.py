import logging

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from db.qdrant_client import QdrantSingleton
from service.embedding_service import EmbeddingService

try:
    from enums import CollectionName
except ImportError:
    from ..enums import CollectionName
from qdrant_client.models import VectorParams, Distance

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
        self.vector_store.add_documents(docs)

    def get_retriever(self, k=4):
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )