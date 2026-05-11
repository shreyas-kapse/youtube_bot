import json

from langchain_core.prompts import PromptTemplate
from langsmith import traceable

from service.BM25_service import BM25Service
from service.LLM_service import LLMService
from service.reranker_service import RerankerService
from service.vector_service import VectorService
from service.query_service import QueryService
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
        self.query_service = QueryService(self.llm)
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

    @traceable
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
        enhanced_docs = self.enhanced_retrieval(query=query, video_id=video_id)
        #BM25
        bm25 = BM25Service(enhanced_docs)
        keyword_docs = bm25.search(query=query, top_k=3)
        
        all_docs = enhanced_docs + keyword_docs * 2

        #Deduplicate
        seen = set()
        combined_docs = []

        for doc in all_docs:
            key = (doc.page_content, doc.metadata.get("timestamp"))
            if key not in seen:
                seen.add(key)
                combined_docs.append(doc)

        #Rerank
        final_docs = RerankerService.rerank(query, combined_docs)

        if not final_docs:
            return {
                "answer": "No relevant content found in this video.",
                "confidence": 0
            }
        top_k_docs = final_docs[:5]

        #Build context
        context = "\n\n".join([
            d.page_content[:300] + f" (ts: {d.metadata.get('timestamp')})"
            for d in top_k_docs
        ])

        #Generate answer
        response = self.llm.invoke(
            self.prompt.format(
                context=context,
                question=query
            )
        )

        return response
    
    @traceable
    def enhanced_retrieval(self, query, video_id):
        rewritten = self.query_service.rewrite_query(query)

        #expand
        queries = self.query_service.expand_query(rewritten)
        all_docs = []

        #retrieve for each query
        for q in queries:
            docs = self.vector_service.search(q, video_id)
            all_docs.extend(docs)

        # deduplicate
        unique_docs = self.deduplicate_docs(all_docs)

        #rerank
        reranked = RerankerService.rerank(query, unique_docs)
        return reranked[:5]
    
    def deduplicate_docs(self, docs):
        seen = set()
        unique = []

        for doc in docs:
            key = doc.page_content

            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique