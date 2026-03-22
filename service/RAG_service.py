from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from service.LLM_service import LLMService
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

        INSTRUCTIONS:
        - Include timestamps in your answer when referencing information
        - Use format like: [05:42]
        - If multiple points → include multiple timestamps
        - Do NOT hallucinate timestamps

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
        
        chain = self.build_chain()

        answer =  chain.invoke(query)
        print("answer : ",answer)
        print("\ntimestamp \n")
        for doc in retrieved_docs:
            start = doc.metadata.get("start_time", 0)
            end = doc.metadata.get("end_time", 0)

            print(
                f"{self.format_timestamp(start)} - {self.format_timestamp(end)}"
            )
            print(
                self.generate_youtube_link(video_id=video_id, start_time=start)
            )
            print("-" * 40)
        
    def format_timestamp(self, seconds):
        return f"{int(seconds//60):02d}:{int(seconds%60):02d}"
    
    def generate_youtube_link(self, video_id, start_time):
        return f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"