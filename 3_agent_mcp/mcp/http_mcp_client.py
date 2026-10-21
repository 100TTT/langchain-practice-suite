#1. 创建一个异步方法来处理MCP客户端的请求
import asyncio
from mcp import stdio_client, ClientSession
from mcp.client.streamable_http import streamable_http_client

url = "http://127.0.0.1:8088/mcp_server"
async def http_run():
    async with streamable_http_client(url=url) as (read_stream, write_stream,_):
        # 建立连接(会话)
        async with ClientSession(read_stream,write_stream) as session:
            # 初始化连接
            await session.initialize()
            # 发送请求获取工具列表
            tools_result = await session.list_tools()
            print(tools_result)
            # 调用工具函数
            res1 = await session.call_tool(tools_result.tools[0].name, {"city": "北京", "date": "2024-12-08"})
            res2 = await session.call_tool(tools_result.tools[1].name, {"city": "北京", "date": "2024-12-08"})
            print(res1)
            print(res2)
if __name__ == "__main__":
    asyncio.run(http_run())
