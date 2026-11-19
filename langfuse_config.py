
import os

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Load environment variables before initializing Langfuse
load_dotenv()

# Validate credentials
required_variables = [
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL"
]

missing_variables = [
    variable
    for variable in required_variables
    if not os.getenv(variable)
]

if missing_variables:
    raise EnvironmentError(
        f"Missing Langfuse variables: {missing_variables}"
    )

# Initialize the Langfuse client
langfuse = get_client()

# Initialize LangChain callback handler
langfuse_handler = CallbackHandler()
