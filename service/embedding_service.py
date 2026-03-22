from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingService:

    _model = None
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = HuggingFaceEmbeddings(
                model_name="intfloat/e5-base-v2"
            )
        return cls._model