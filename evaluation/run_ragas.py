import json

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv                           

load_dotenv()                         
                                  
# -----------------------------
# Local evaluator LLM
# -----------------------------
# llm = ChatGoogleGenerativeAI(
#     model="Gemini 2.5 Flash",
#     temperature=0
# )
llm=ChatOllama(
    model="phi3",
    temperature=0
)

# -----------------------------
# Embedding model
# -----------------------------
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

ragas_llm = LangchainLLMWrapper(llm)

ragas_embeddings = LangchainEmbeddingsWrapper(
    embeddings
)


with open("evaluation/eval_dataset.json", "r") as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    "question": [x["question"] for x in data],
    "answer": [x["answer"] for x in data],
    "contexts": [x["contexts"] for x in data],
    "ground_truth": [x["ground_truth"] for x in data],
})

# -----------------------------
# Prevent timeout issues
# -----------------------------
run_config = RunConfig(
    max_workers=1,
    timeout=300
)


print("\nStarting RAGAS evaluation...\n")

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy
        # context_precision,
        # context_recall,
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=run_config
)


df = result.to_pandas()

print("\nDetailed Results:\n")
print(df)

avg_scores = df.mean(numeric_only=True)

print("\nAverage Scores:\n")
print(avg_scores)


with open("evaluation/ragas_results.json", "w") as f:
    json.dump(
        {
            "average_scores": avg_scores.to_dict(),
            "detailed_results": df.to_dict(orient="records")
        },
        f,
        indent=4
    )

print("\nRAGAS evaluation completed.")