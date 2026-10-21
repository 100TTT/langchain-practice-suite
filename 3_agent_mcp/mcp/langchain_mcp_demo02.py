#目标: 给用户查询12306的火车票情况
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

#创建MCP客户端
mcp_client = MultiServerMCPClient(
    {
        "12306-mcp": {
          "transport": "streamable_http",
          "url": "https://mcp.api-inference.modelscope.net/90b1bbbfb2fb49/mcp"
        }
    }
)

async def main():
    question = "请帮我查询一下2026年3月23日从北京到深圳的高铁票情况？帮我以Markdown的方式列出有哪些车次，几点出发，几点到达，票价多少？"

    tools = await mcp_client.get_tools()

    agent = create_agent(
        model="gpt-4o-mini",
        system_prompt="你是一名12306火车票助手",
        tools=tools
    )

    res = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    print(res['messages'][-1].content)
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())