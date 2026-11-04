
import os

from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings

load_dotenv()


class OpenRouterEmbeddings(Embeddings):

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=60.0,
            max_retries=2
        )

        self.model = "openai/text-embedding-3-small"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        sorted_data = sorted(
            response.data,
            key=lambda item: item.index
        )

        return [item.embedding for item in sorted_data]

    def embed_query(self, text: str) -> list[float]:

        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        return response.data[0].embedding
