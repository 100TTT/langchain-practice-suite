import os

# 在导入任何 langchain 之前，强行设置环境变量！
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True) # 这里加上 override=True


#1. 定义工具函数
class GetWeatherArgs(BaseModel):
    city: str = Field(description="城市名称,例如:深圳市")
    date: str = Field(description="日期，格式为YYYY-MM-DD")

@tool(description="查询某个城市在指定日期下的天气信息", args_schema=GetWeatherArgs)
def get_weather(city: str, date: str):
    # 这里我们模拟一个天气信息的返回
    return f"{city}在{date}的天气是晴朗，温度25度。"

#2. 使用大模型对象创建agent
qwen_llm = ChatOpenAI(
    model="qwen3.8-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
agent = create_agent(
    model=qwen_llm,  # 这里传对象，千万别传字符串
    tools=[get_weather],
    system_prompt="你是位助手，需要调用工具来帮助用户。"
)
#3. 调用大模型
res = agent.invoke({"messages": [{"role": "user", "content": "今天北京的天气怎么样？"}]})

print(res)

