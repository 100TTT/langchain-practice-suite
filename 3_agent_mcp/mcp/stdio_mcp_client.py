#1. 创建一个异步方法来处理MCP客户端的请求
import asyncio

from mcp import stdio_client, StdioServerParameters, ClientSession

stdio_server_params = StdioServerParameters(
    command=r"D:\workspace\pycharm\BJ_1208\langchain_1208_demo\.venv\Scripts\python.exe",
    args=[r"D:\workspace\pycharm\BJ_1208\langchain_1208_demo\agents\mcp\stdio_mcp_server.py"]
)
async def stdio_run():
    async with stdio_client(stdio_server_params) as (read_stream, write_stream):
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
    asyncio.run(stdio_run())
