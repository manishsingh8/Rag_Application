import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3
)

def clean_question(question):
    return question.strip()

cleaning_runnable = RunnableLambda(clean_question)


prompt = ChatPromptTemplate.from_template(
    """
    You are an experienced database engineer.

    Explain the following database concept
    in three sections: Definition, Example, and Use Case.

    Question: {question}
    Level:{level}
    """
)
parser = StrOutputParser()

chain = cleaning_runnable | prompt | llm | parser

response = chain.invoke(
    "   What is MongoDB?   "
)

print(response)