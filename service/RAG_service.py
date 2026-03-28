import json
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from service.BM25_service import BM25Service
from service.LLM_service import LLMService
from service.reranker_service import RerankerService
from service.vector_service import VectorService

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        start = doc.metadata.get("start_time", 0)
        time = f"{int(start//60):02d}:{int(start%60):02d}"
        formatted_docs.append(
            f"[{time}] {doc.page_content}"
        )
    return "\n\n".join(formatted_docs)

class RAGService:

    def __init__(self):
        self.vector_service = VectorService()
        self.retriever = self.vector_service.get_retriever()
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
        ]
        }}

        INSTRUCTIONS:
        - Break your answer into multiple sentences
        - Each sentence MUST have exactly ONE relevant timestamp
        - Extract the timestamp from the context
        - Rewrite the content in your own words (DO NOT copy sentences exactly)
        - Make sentences simple, clear, and easy to understand
        - Preserve the original meaning from the context
        - Do NOT add information not present in the context
        - Do NOT group multiple sentences under one timestamp
        - Do NOT return anything outside the JSON format

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
            input_variables=["context", "question"]
        )

        self.parser = StrOutputParser()

    def build_chain(self):
        parallel_chain = RunnableParallel(
            {
                "context": self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
        )

        return parallel_chain | self.prompt | self.llm | self.parser

    def ask(self, query: str, video_id: str):

        query = f"query: {query}"
        vector_docs = self.retriever.invoke(query)
        
        bm25 = BM25Service(vector_docs)
        keyword_docs = bm25.search(query=query, top_k=3)
        
        combined_docs = list({doc.page_content: doc for doc in vector_docs + keyword_docs}.values())
        
        reranked_docs = RerankerService.rerank(query=query.replace("query: ",""), docs=combined_docs)
        
        top_k = reranked_docs[:5]
        context = format_docs(top_k)
        
        answer =  self.llm.invoke(
            self.prompt.format(
                context=context,
                question= query
            )
        )
        
        json_response = self.parse_output(answer.text)
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        for seg in json_response["segments"]:
            ts = seg["timestamp"].strip("[]")  
            m, s = map(int, ts.split(":"))
            seg["url"] = f"{video_url}&t={m*60 + s}s"

        json_response["query"] = query
        return json.dumps(json_response)
    
    def parse_output(cls, output):
        try:
            return json.loads(output)
        except:
            return {"answer": output, "timestamps": []}