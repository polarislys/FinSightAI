import os
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient

# 1. 加载环境变量
load_dotenv()

def test_llm():
    print("--- 正在测试大模型连接 (Qwen via SiliconFlow) ---")
    try:
        client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL")
        )
        
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "user", "content": "你好，请介绍一下你自己，并简述RAG是什么？"}
            ],
            stream=False
        )
        print(f"✅ LLM 连接成功！回复内容：\n{response.choices[0].message.content}\n")
    except Exception as e:
        print(f"❌ LLM 连接失败: {e}")

def test_milvus():
    print("--- 正在测试 Milvus Lite 向量库 ---")
    try:
        # 在当前目录下生成一个名为 finsight.db 的轻量级数据库文件
        client = MilvusClient("./finsight.db")
        
        # 创建一个测试集合
        collection_name = "demo_collection"
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            
        client.create_collection(
            collection_name=collection_name,
            dimension=1024  # 假设向量维度，测试用
        )
        
        print(f"✅ Milvus Lite 启动成功！已创建数据库文件: ./finsight.db")
        print(f"✅ 已创建测试集合: {collection_name}")
    except Exception as e:
        print(f"❌ Milvus 测试失败: {e}")

if __name__ == "__main__":
    test_llm()
    test_milvus()