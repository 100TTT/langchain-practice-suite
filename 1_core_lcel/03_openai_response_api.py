#1. 读取环境变量中的信息
from openai import OpenAI
from openai.types.responses import WebSearchPreviewToolParam

#2. 创建对象
llm = OpenAI()

#3. 调用对象方法
res = llm.responses.create(
    model="gpt-4o-mini",
    input="2026年3月20日中国国内有哪些新闻,给我总结一下",
    tools=[WebSearchPreviewToolParam(type="web_search_preview")]
)
#4. 输出结果
print(res)