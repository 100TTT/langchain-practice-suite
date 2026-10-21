from mcp.server.fastmcp import FastMCP
#1. 创建MCP服务器实例
mcp_server = FastMCP(name="mcp_server",port=8088,streamable_http_path="/mcp_server")

#2. 定义mcp的工具函数
@mcp_server.tool()
def get_weather(city: str, date: str):
    # 这里我们模拟一个天气信息的返回
    return f"{city}在{date}的天气是晴朗"

@mcp_server.tool()
def get_temperature(city: str, date: str):
    # 这里我们模拟一个天气信息的返回
    return f"{city}在{date}的温度是25度。"

#3. 启动MCP服务器
if __name__ == "__main__":
    mcp_server.run(transport="streamable-http")
