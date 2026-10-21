import logging

from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai"
)
logging.basicConfig(level=logging.DEBUG)
res = llm.batch(
    inputs=[
        "什么是Langchain?",
        "什么是RAG?",
        "什么是Agent?"
    ]
)

for message in res:
    print(message.content)

