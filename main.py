import logging

from service.RAG_service import RAGService
from service.ingestion_service import IngestionService
from utils.logger import setup_logger

logger = logging.getLogger(__name__)

def main():

    setup_logger()
    video_id = "hBMoPUAeLnY"
    
    ingestion = IngestionService()
    rag = RAGService()
    ingestion.ingest_video(video_id)

    answer = rag.ask(query = "summarize What donald trump saying in this video.", video_id= video_id)
    print(answer)

if __name__ == "__main__":
    main()