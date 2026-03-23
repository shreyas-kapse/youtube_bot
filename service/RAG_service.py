import json
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

        Use ONLY the provided context.

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
        - Do NOT group multiple sentences under one timestamp
        - Do NOT return a separate timestamps list
        - Do NOT add extra text outside JSON

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
        retrieved_docs = self.retriever.invoke(query)
        
        reranked_docs = RerankerService.rerank(query=query, docs=retrieved_docs)
        
        top_k = reranked_docs[:3]
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
        
        return json.dumps(json_response)
    
    def parse_output(cls, output):
        try:
            return json.loads(output)
        except:
            return {"answer": output, "timestamps": []}