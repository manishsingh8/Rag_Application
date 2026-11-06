from typing import TypedDict
from langchain_core.documents import Document

from vector_store import get_retriever
from rag_chain import prompt, structured_llm, format_documents
from langgraph.graph import StateGraph, START, END


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    citations: list[str]
    evidence_status: str
    retry_count: int
    is_valid: bool
    validation_errors: list[str]


retriever = get_retriever()


def retrieve_documents(state: RAGState):
    print("\n--- Retrieving Documents ---")

    question = state["question"]

    documents = retriever.invoke(question)

    print(f"Retrieved {len(documents)} documents")

    return {
        "documents": documents
    }

def generate_answer(state: RAGState):

    print("\n--- Generating Answer ---")

    question = state["question"]
    documents = state["documents"]

    # Format retrieved documents into context
    context = format_documents(documents)

    # Prepare input for the prompt
    prompt_input = {
        "question": question,
        "context": context
    }
    
    # Generate structured response
    response = (prompt | structured_llm).invoke(prompt_input)
    print("Answer generated successfully!")

    return {
        "answer": response.answer,
        "citations": response.citations,
        "evidence_status": response.evidence_status
    }

def validate_answer(state: RAGState):

    print("\n--- Validating Answer ---")

    documents = state["documents"]
    citations = state["citations"]
    evidence_status = state["evidence_status"]

    # Build valid citation references from retrieved documents
    valid_citations = {
        f"Source {index}"
        for index in range(1, len(documents) + 1)
    }

    # Normalize citations to handle both "Source 1" and "[Source 1]"
    normalized_citations = {
        citation.strip().strip("[]")
        for citation in citations
    }

    errors = []

    # Check whether citations refer to retrieved documents
    invalid_citations = normalized_citations - valid_citations

    if invalid_citations:
        errors.append(
            f"Invalid citations: {list(invalid_citations)}"
        )

    # Validate evidence status
    if evidence_status == "supported" and not citations:
        errors.append("Supported answer must contain citations.")

    if evidence_status == "insufficient_evidence" and citations:
        errors.append(
            "Insufficient-evidence answer should not contain citations."
        )

    if evidence_status not in {
        "supported",
        "insufficient_evidence"
    }:
        errors.append("Invalid evidence status.")

    is_valid = len(errors) == 0

    print("Validation successful!" if is_valid else "Validation failed!")

    return {
        "is_valid": is_valid,
        "validation_errors": errors
    }

def route_after_validation(state: RAGState):

    print("\n--- Routing Decision ---")

    if state["is_valid"]:
        print("Answer is valid. Finishing workflow.")
        return "valid"

    print("Answer failed validation.")
    return "invalid"

def handle_invalid_answer(state: RAGState):

    print("\n--- Handling Invalid Answer ---")

    return {
        "answer": (
            "I couldn't generate a response that passed "
            "citation validation. Please try again."
        ),
        "citations": [],
        "evidence_status": "insufficient_evidence"
    }



# Create graph
graph_builder = StateGraph(RAGState)

# Register nodes
graph_builder.add_node("retrieve_documents", retrieve_documents)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("validate_answer", validate_answer)
graph_builder.add_node("handle_invalid_answer", handle_invalid_answer)

# Normal edges
graph_builder.add_edge(START, "retrieve_documents")

graph_builder.add_edge(
    "retrieve_documents",
    "generate_answer"
)

graph_builder.add_edge(
    "generate_answer",
    "validate_answer"
)

# Conditional edge
graph_builder.add_conditional_edges(
    "validate_answer",
    route_after_validation,
    {
        "valid": END,
        "invalid": "handle_invalid_answer"
    }
)

# Fallback goes to END
graph_builder.add_edge("handle_invalid_answer", END)

# Compile graph
rag_graph = graph_builder.compile()



if __name__ == "__main__":

    print("LangGraph RAG Application Started!")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        initial_state = {
            "question": question,
            "documents": [],
            "answer": "",
            "citations": [],
            "evidence_status": "",
            "retry_count": 0,
            "is_valid": False,
            "validation_errors": []
        }

        try:
            result = rag_graph.invoke(initial_state)

            print("\n--- Assistant Response ---")
            print("Answer:", result["answer"])
            print("Citations:", result["citations"])
            print("Evidence Status:", result["evidence_status"])

        except Exception as exc:
            print("\nError:", exc)