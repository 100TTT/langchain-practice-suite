import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# 调用MCPServer获取工具
client = MultiServerMCPClient(
    {
        "weather": {
            "url": "http://localhost:8088/mcp_server",
            "transport": "http",
        }
    }
)


async def main():
    # 调用MCPServer获取工具
    tools = await client.get_tools()
    #1. 创建agent
    agent = create_agent(
        model="gpt-4o-mini",
        system_prompt="你是位助手，需要调用工具来帮助用户。",
        tools=tools
    )

    # 2. 调用大模型
    res = await agent.ainvoke({"messages": [{"role": "user", "content": "2026年3月23日北京的天气怎么样？我适合穿什么衣服"}]},config={"configurable":{"thread_id":"1"}})

    print(res)

if __name__ == "__main__":
    asyncio.run(main())