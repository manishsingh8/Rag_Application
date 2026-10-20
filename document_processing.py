
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Locate the document
file_path = Path("data/database.txt")


# 2. Load the document
loader = TextLoader(
    str(file_path),
    encoding="utf-8",
)

documents = loader.load()

print(f"Number of loaded documents: {len(documents)}")
print("\nOriginal document:")
print(documents[0].page_content)

print("\nMetadata:")
print(documents[0].metadata)


# 3. Configure the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
)


# 4. Split the documents
chunks = text_splitter.split_documents(documents)

print(f"\nNumber of chunks: {len(chunks)}")


# 5. Inspect every chunk
for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
