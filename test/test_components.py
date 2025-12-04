# 创建一个简单的测试脚本 test_components.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient

load_dotenv()

print("🔍 测试各组件...")

# 测试 OpenAI 连接
try:
    client = OpenAI(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL")
    )
    print("✅ OpenAI 客户端初始化成功")
except Exception as e:
    print(f"❌ OpenAI 客户端失败: {e}")

# 测试 Milvus
try:
    milvus_client = MilvusClient("./test.db")
    print("✅ Milvus 客户端初始化成功")
except Exception as e:
    print(f"❌ Milvus 客户端失败: {e}")

# 测试 Text Splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    test_text = "这是一个测试文本。用来验证文本切分功能是否正常工作。"
    chunks = splitter.split_text(test_text)
    print(f"✅ 文本切分成功，生成 {len(chunks)} 个片段")
except Exception as e:
    print(f"❌ 文本切分失败: {e}")