import logging
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

# 目标: 用OpenAI模型、以及Google的模型对诗进行分析，然后将两个分析结果交给DeepSeek的模型进行总结评价
# 1. 创建第一条调用链
# 1.1 定义提示词模板
paragraph_template_1 = ChatPromptTemplate([("human", "对这首诗{poem}做一下赏析，分析它蕴含的含义")])
# 1.2 定义模型对象
openai_llm = ChatOpenAI(
    model="gpt-4o-mini"
)
# 1.3 定义解析器
str_output_parser = StrOutputParser()
# 1.4 组成第一条链
openai_chain = paragraph_template_1 | openai_llm | str_output_parser

# 2. 创建第二条调用链
# 2.1 定义提示词模板（跟上面共用同一个模板）
# 2.2 定义模型对象
gemini_llm = ChatOpenAI(
    model="gemini-2.5-flash-lite"
)
# 2.3 定义解析器（跟上面用同一个）
# 2.4 组成第一条链
gemini_chain = paragraph_template_1 | gemini_llm | str_output_parser

# 3. 创建第三条调用链
# 3.1 定义提示词模板
summary_template = ChatPromptTemplate(
    [("human", "这两种赏析，第一种：{paragraph_1}，第二种：{paragraph_2}，哪个更好，为什么")])
# 3.2 定义模型对象
deepseek_llm = ChatOpenAI(
    model="deepseek-chat"
)
# 3.3 定义解析器（跟上面用同一个）
# 3.4 组成第三条链
deepseek_chain = summary_template | deepseek_llm | str_output_parser

# chains = RunnableParallel(
#     paragraph_1=openai_chain,
#     paragraph_2=gemini_chain,
# ) | deepseek_chain

chains = {
             "paragraph_1": openai_chain,
             "paragraph_2": gemini_chain,
         } | deepseek_chain

# 执行整个调用链
poem = """
    菩提本无树，
    明镜亦非台，
    本来无一物，
    何处惹尘埃。
    """
res = chains.invoke({
    "poem": poem
})

print(res)
