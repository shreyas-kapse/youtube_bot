import json

from langchain_core.prompts import PromptTemplate

from service.BM25_service import BM25Service
from service.LLM_service import LLMService
from service.reranker_service import RerankerService
from service.vector_service import VectorService

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        start = doc.metadata.get("start_time", 0)
        time = f"{int(start//60):02d}:{int(start%60):02d}"
        formatted_docs.append(f"[{time}] {doc.page_content}")
    return "\n\n".join(formatted_docs)

class RAGService:

    def __init__(self):
        self.vector_service = VectorService()
        self.llm = LLMService.get_llm()
        self.prompt = PromptTemplate(
            template="""
        You are an AI assistant answering questions from a YouTube video.

        Use ONLY the provided context as your source of information.

        Each context chunk has a timestamp like [MM:SS].

        Return STRICT JSON in this format:
        {{"segments": [
            {{
            "sentence": "string",
            "timestamp": "MM:SS"
            }}
        ]}}

        INSTRUCTIONS:
        - Break your answer into multiple sentences
        - Each sentence MUST have exactly ONE relevant timestamp
        - Extract the timestamp from the context
        - Rewrite the content in your own words
        - Keep it simple and clear
        - Do NOT add information not in context
        - Do NOT return anything outside JSON

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"])

    def get_retriever(self, video_id: str, k=10):
        return self.vector_service.get_retriever(
            video_id=video_id,
            k=k
        )

    def parse_output(self, output):
        try:
            return json.loads(output)
        except:
            return {"segments": []}


    def ask(self, query: str, video_id: str):

        retriever = self.get_retriever(video_id=video_id)
        vector_docs = retriever.invoke(query)

        bm25 = BM25Service(vector_docs)
        keyword_docs = bm25.search(query=query, top_k=3)

        combined_docs = list({
            doc.page_content: doc
            for doc in (vector_docs + keyword_docs)
        }.values())

        reranked_docs = RerankerService.rerank(
            query=query,
            docs=combined_docs
        )

        top_k_docs = reranked_docs[:5]
        context = format_docs(top_k_docs)

        response = self.llm.invoke(
            self.prompt.format(
                context=context,
                question=query
            )
        )

        json_response = self.parse_output(response.text)
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        for seg in json_response.get("segments", []):
            ts = seg["timestamp"].strip("[]")

            try:
                m, s = map(int, ts.split(":"))
                seconds = m * 60 + s
                seg["url"] = f"{video_url}&t={seconds}s"
            except:
                seg["url"] = video_url

        json_response["query"] = query
        return json_response