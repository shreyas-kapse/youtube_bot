import json
from service.qa_service import QaService
from service.RAG_service import RAGService
from service.ingestion_service import IngestionService

qa_service = QaService(
    rag_service=RAGService(),
    ingestion_service=IngestionService(),
)

with open("evaluation/eval_dataset.json", "r") as f:
    dataset = json.load(f)


for item in dataset:

    question = item["question"]
    video_id = item["video_id"]

    print(f"\nProcessing: {question}")

    try:

        # Generate answer
        response = qa_service.answer_question(
            query=question,
            video_id=video_id
        )

        answer = response.content

        # Retrieve docs separately
        docs = qa_service.rag_service.enhanced_retrieval(
            query=question,
            video_id=video_id
        )

        contexts = [
            doc.page_content
            for doc in docs[:5]
        ]

        item["answer"] = answer
        item["contexts"] = contexts

        print("Done")

    except Exception as e:

        print(f"Error: {e}")

        item["answer"] = ""
        item["contexts"] = []


with open("evaluation/eval_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print("\nEvaluation dataset updated.")