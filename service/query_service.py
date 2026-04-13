class QueryService:
    def __init__(self, llm):
        self.llm = llm

    def rewrite_query(self, query: str) -> str:
        prompt = f"""
        Rewrite the query to be more specific and clear for semantic search.
        Original: {query}
        Rewritten:
        """

        response = self.llm.invoke(prompt)
        return response.content
    
    def expand_query(self, query: str) -> list[str]:
        prompt = f"""
        Generate 3 alternative search queries for:
        "{query}"
        Return as a list.
        """
        response = self.llm.invoke(prompt)

        text = response.content  
        queries = [
            q.strip("- ").strip()
            for q in text.split("\n")
            if q.strip()
        ]

        return list(set([query] + queries))