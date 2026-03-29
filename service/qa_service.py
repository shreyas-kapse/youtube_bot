class QaService:
    def __init__(self, rag_service, ingestion_service):
        self.rag_service = rag_service
        self.ingestion_service= ingestion_service
    
    def process_video_embeddings(self, video_id):
        self.video_id = video_id
        self.ingestion_service.ingest_video(video_id)
        
        
    def answer_question(self, query:str, video_id:str):
        answer = self.rag_service.ask(query=query, video_id=video_id)
        return answer