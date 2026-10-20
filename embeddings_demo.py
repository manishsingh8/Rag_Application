
import os
import math

from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

# 1. Connect to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

EMBEDDING_MODEL = "openai/text-embedding-3-small"

# 2. Load the document
loader = TextLoader("data/database.txt", encoding="utf-8")
documents = loader.load()

# 3. Split it into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)
texts = [chunk.page_content for chunk in chunks]

if not texts:
    raise ValueError("No text chunks were created.")

# 4. Generate embeddings for all chunks in one request
response = client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=texts,
)

# Keep each vector aligned with its corresponding chunk
embeddings = [
    item.embedding
    for item in sorted(response.data, key=lambda item: item.index)
]

print(f"Chunks created: {len(chunks)}")
print(f"Embedding dimensions: {len(embeddings[0])}")
print(f"First vector (first 8 numbers): {embeddings[0][:8]}")


# 5. Calculate cosine similarity
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# 6. Embed the user's question
question = "How can I make database queries faster?"

query_response = client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=question,
)

query_embedding = query_response.data[0].embedding

# 7. Rank chunks by similarity to the question
ranked_results = []

for chunk, embedding in zip(chunks, embeddings):
    score = cosine_similarity(query_embedding, embedding)

    ranked_results.append({
        "text": chunk.page_content,
        "score": score,
        "metadata": chunk.metadata,
    })

ranked_results.sort(
    key=lambda result: result["score"],
    reverse=True,
)

print("\nMost relevant chunks:")

for result in ranked_results[:3]:
    print(f"\nSimilarity: {result['score']:.4f}")
    print(f"Text: {result['text']}")
    print(f"Source: {result['metadata']}")
