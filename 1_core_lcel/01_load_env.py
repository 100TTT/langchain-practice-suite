from dotenv import load_dotenv

#1. 加载环境变量文件
load_dotenv()

#2. 从环境变量中读取内容
import os
base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")

print(base_url)
print(api_key)