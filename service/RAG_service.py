from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from service.LLM_service import LLMService
from service.vector_service import VectorService


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAGService:

    def __init__(self):
        self.vector_service = VectorService()
        self.retriever = self.vector_service.get_retriever()
        self.llm = LLMService.get_llm()
        
        self.prompt = PromptTemplate(
            template="""
                You are a helpful assistant.
                Answer ONLY using the provided context.
                Context:
                {context}
                Question:
                {question}
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

    def ask(self, question: str):

        question = f"query: {question}"
        self.retriever.invoke(question)
        
        chain = self.build_chain()

        return chain.invoke(question)