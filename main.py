import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Explain technical concepts simply."
    }
]
while True:
    query = input("\nYou: ")

    if query.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": query
    })

    stream = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
        stream=True
    )

    print("AI: ", end="", flush=True)

    assistant_message = ""

    for chunk in stream:
        content = chunk.choices[0].delta.content

        if content:
            print(content, end="", flush=True)
            assistant_message += content

    print()

    messages.append({
        "role": "assistant",
        "content": assistant_message
    })

    print("AI:", chunk.choices[0].delta.content)