from sentence_transformers import CrossEncoder


class RerankerService:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        return cls._model

    @classmethod
    def rerank(cls, query: str, docs: list):
        model = cls.get_model()
        pairs = [(query, doc.page_content) for doc in docs]
        
        scores = model.predict(pairs)
        scored_docs = list(zip(docs, scores))

        ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked]