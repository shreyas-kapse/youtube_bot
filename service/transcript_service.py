from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

class TranscriptService:
    
    def get_video_transcript(cls, video_id: str) -> str:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript


    def transcript_to_documents(cls, transcript:str) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.create_documents([transcript])
        for doc in docs:
            doc.page_content = f"passage: {doc.page_content}"

        return docs
