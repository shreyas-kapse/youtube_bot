from langchain_core.documents import Document

from youtube_transcript_api import YouTubeTranscriptApi

class TranscriptService:
    
    def get_video_transcript(cls, video_id: str) -> str:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id, languages=["en"])
        transcript=[]
        for chunk in transcript_list:
            transcript.append({
                "text":chunk.text,
                "start": chunk.start,
                "duration": chunk.duration,
                "end": chunk.start + chunk.duration
            })
        return transcript


    def transcript_to_documents(cls, transcript: list[dict], video_id: str) -> list[Document]:
        documents = []

        current_text = ""
        start_time = None

        for segment in transcript:
            if start_time is None:
                start_time = segment["start"]

            current_text += " " + segment["text"]

            if len(current_text.split()) >= 200:
                documents.append(
                    Document(
                        page_content=f"passage: {current_text.strip()}",
                        metadata={
                            "video_id": video_id,
                            "start_time": start_time,
                            "end_time": segment["end"]
                        }
                    )
                )
                current_text = ""
                start_time = None

        if current_text:
            documents.append(
                Document(
                    page_content=f"passage: {current_text.strip()}",
                    metadata={
                        "video_id": video_id,
                        "start_time": start_time,
                        "end_time": transcript[-1]["end"]
                    }
                )
            )

        return documents

