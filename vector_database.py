
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load environment variables
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")


# 2. Connect to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

EMBEDDING_MODEL = "openai/text-embedding-3-small"


def generate_embeddings(texts):
    """Convert one or more text strings into embedding vectors."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [
        item.embedding
        for item in sorted(response.data, key=lambda item: item.index)
    ]


# 3. Load and split the document
file_path = Path("data/database.txt")

loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)

if not chunks:
    raise ValueError("No document chunks were created.")


# 4. Create persistent Chroma storage
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="database_docs",
    metadata={"hnsw:space": "cosine"},
)


# 5. Replace existing chunks from this source.
# This keeps reruns from leaving outdated chunks behind.
source = str(file_path)

existing = collection.get(
    where={"source": source},
)

if existing["ids"]:
    collection.delete(ids=existing["ids"])


# 6. Generate embeddings and store the chunks
texts = [chunk.page_content for chunk in chunks]

embeddings = generate_embeddings(texts)

ids = [
    f"database-chunk-{index}"
    for index in range(len(chunks))
]

metadatas = [
    {
        **chunk.metadata,
        "source": source,
        "chunk_index": index,
    }
    for index, chunk in enumerate(chunks)
]

collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
)

print(f"Stored {len(chunks)} chunks in Chroma.")
print(f"Total chunks in collection: {collection.count()}")


# 7. Search the stored vectors
question = "How can I make database queries faster?"

query_embedding = generate_embeddings([question])[0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(3, collection.count()),
    include=["documents", "metadatas", "distances"],
)

print("\nMost relevant results:")

for index, text in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][index]
    distance = results["distances"][0][index]

    print(f"\n--- Result {index + 1} ---")
    print(f"Text: {text}")
    print(f"Source: {metadata['source']}")
    print(f"Chunk index: {metadata['chunk_index']}")
    print(f"Cosine distance: {distance:.4f}")
