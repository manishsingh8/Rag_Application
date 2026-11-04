from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Initialize Langfuse
langfuse = get_client()

# Initialize LangChain callback handler
langfuse_handler = CallbackHandler()

print("Langfuse initialized successfully!")