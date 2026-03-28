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

    while True:
        query = input("press -1 to exit the application\n ")
        if  str(query).strip() =='-1':
            break
        answer = rag.ask(query=query, video_id=video_id)
        print(answer)
    
if __name__ == "__main__":
    main()