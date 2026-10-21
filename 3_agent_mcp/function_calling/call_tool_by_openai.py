import json

from openai import OpenAI
llm = OpenAI()

# 定义一个获取天气信息的工具函数
def get_weather(city:str,date:str):
    # 这里我们模拟一个天气信息的返回
    return f"{city}在{date}的天气是晴朗，温度25度。"

# 1. 通过JSON结构定义工具，包括工具名称，描述，参数等
tools = [
      {
            "type": "function",
            "function": {
                  "name": "get_weather",
                  "description": "查询某个城市在指定日期下的天气信息.",
                  "parameters": {
                        "type": "object",
                        "properties": {
                              "city": {
                                    "type": "string",
                                    "description": "城市名称, e.g. 北京市",
                              },
                              "date" :{
                                    "type":"string",
                                    "description":"想要查询的天气的日期, e.g. 2023-12-25"
                              }
                        },
                        "required": ["city","date"],
                        "additionalProperties": False,
                  },
                  "strict": True,
            },
      },
]

messages = [
        {"role": "user", "content": "请告诉我北京在2026年3月23号的天气情况，并且告诉我在这个天气下适合穿什么样的衣服？"}
]
# 调用大模型
res = llm.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)
# 取出大模型输出结果中的tool_calls的信息
tool_calls = res.choices[0].message.tool_calls
if tool_calls:
    function = tool_calls[0].function
    id = tool_calls[0].id
    tool_name = function.name
    arguments = function.arguments
    obj = json.loads(arguments)
    if tool_name == "get_weather":
        weather = get_weather(obj["city"],obj["date"])

        # 将第一次调用工具时候大模型返回的messages拼接进去
        messages.append(res.choices[0].message)

        # 将调用工具查询到的天气添加到messages里面
        messages.append({"role":"tool","tool_call_id":id,"content":json.dumps({"weather":weather})})
        # 第二次调用大模型
        res = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        print(res)
else:
    print(res)
