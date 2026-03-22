from service.transcript_service import TranscriptService
from service.vector_service import VectorService


class IngestionService:

    def __init__(self):
        self.vector_service = VectorService()
        self.transcript_service = TranscriptService()
        
    def ingest_video(self, video_id):
        transcript = self.transcript_service.get_video_transcript(video_id)
        docs = self.transcript_service.transcript_to_documents(transcript= transcript, video_id= video_id)
        self.vector_service.add_documents(docs=docs)