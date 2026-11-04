from langchain_chroma import Chroma
from openrouter_embeddings import OpenRouterEmbeddings

embeddings = OpenRouterEmbeddings()

vector_store = Chroma(
    collection_name="database_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

def get_retriever():
    return vector_store.as_retriever(
        search_kwargs={"k": 3}
    )