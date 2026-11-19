
from vector_store import get_retriever

retriever = get_retriever()

question = "How can I improve MongoDB query performance?"

documents = retriever.invoke(question)

print(f"\nRetrieved {len(documents)} documents\n")

for index, document in enumerate(documents, start=1):

    print(f"Source {index}")
    print("Content:", document.page_content)
    print("Metadata:", document.metadata)
    print("-" * 50)
