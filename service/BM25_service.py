from rank_bm25 import BM25Okapi


class BM25Service:

    def __init__(self, docs):
        self.docs = docs
        self.corpus = [doc.page_content.split() for doc in docs]
        self.model = BM25Okapi(self.corpus)

    def search(self, query, top_k=5):
        tokenized_query = query.split()
        scores = self.model.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [doc for doc, _ in ranked[:top_k]]