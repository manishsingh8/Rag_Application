
from langchain_core.documents import Document

from rag_graph import evaluate_answer


question = "What is MongoDB?"

documents = [
    Document(
        page_content=(
            "MongoDB is a NoSQL database that stores "
            "data in flexible JSON-like documents."
        )
    )
]

answer = (
    "MongoDB is a relational database that stores "
    "data in tables."
)

evaluation = evaluate_answer(
    question,
    answer,
    documents
)

print("\n--- LLM Judge Evaluation ---")
print("Faithfulness:", evaluation.faithfulness_score)
print("Answer Relevance:", evaluation.answer_relevance_score)
print("Unsupported Claims:", evaluation.unsupported_claims)
print("Reasoning:", evaluation.reasoning)
