import logging
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class QdrantSingleton:

    _instance = None
    _client = None

    def __new__(cls):
        
        if cls._instance is None:
            logger.info("Creating new Qdrant client")
            
            cls._instance = super(QdrantSingleton, cls).__new__(cls)
            cls._client = QdrantClient(
                host="localhost",
                port=6333
            )
        return cls._instance

    def get_client(self):
        return self._client