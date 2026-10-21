#定义一个工具函数
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class GetWeatherArgs(BaseModel):
      city: str = Field(description="城市名称,例如:深圳市")
      date: str = Field(description="日期，格式为YYYY-MM-DD")

@tool(description="查询某个城市在指定日期下的天气信息",args_schema=GetWeatherArgs)
def get_weather(city:str,date:str):
    # 这里我们模拟一个天气信息的返回
    return f"{city}在{date}的天气是晴朗，温度25度。"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

# 让大模型对象绑定工具
call_tool_llm = llm.bind_tools([get_weather])

# 调用大模型
message_list = [
    HumanMessage(content="请告诉我北京在2026年3月23号的天气情况，并且告诉我在这个天气下适合穿什么样的衣服？")
]
res = call_tool_llm.invoke(
    input=message_list
)

# 将res中返回的AIMessage添加到message_list
message_list.append(res)

# 解析出tool_calls
tool_name = res.tool_calls[0]["name"]
args = res.tool_calls[0]["args"]
tool_call_id = res.tool_calls[0]["id"]

# 调用工具
call_tool_result = get_weather.invoke(args)

# 构建ToolMessage
tool_message = ToolMessage(content = call_tool_result,tool_call_id=tool_call_id)
message_list.append(tool_message)

# 第二次调用大模型
res2 = call_tool_llm.invoke(
    input=message_list
)

print(res2)