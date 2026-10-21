from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai"
)

res = llm.stream(
    input=[
        SystemMessage(content="你是一名资深狗仔队"),
        HumanMessage(content="周杰伦和田馥甄有没有谈过恋爱?")
    ]
)

for message in res:
    print(message)
    #print(message.content)

