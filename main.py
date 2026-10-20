
import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")


# 1. Configure the model
model = ChatOpenRouter(
    model="openrouter/free",
    api_key=api_key,
    temperature=0,
)


# 2. Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Explain technical concepts simply, with examples."
    ),
    (
        "human",
        "{question}"
    ),
])


# 3. Define the output parser
parser = StrOutputParser()


# 4. Connect the components
chain = prompt | model | parser


# 5. Run the chat loop
while True:
    query = input("\nYou: ").strip()

    if query.lower() == "exit":
        break

    if not query:
        continue

    try:
        answer = chain.invoke({
            "question": query
        })

        print("\nAI:", answer)

    except Exception as error:
        print(f"\nRequest failed: {error}")
