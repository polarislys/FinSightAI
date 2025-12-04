import os
from dotenv import load_dotenv

load_dotenv()

print("🔍 测试 OpenAI 导入...")

try:
    from openai import OpenAI
    print("✅ OpenAI 导入成功")
    
    print("🔍 测试 OpenAI 客户端初始化...")
    client = OpenAI(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL")
    )
    print("✅ OpenAI 客户端初始化成功")
    
    print("🔍 测试简单 API 调用...")
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": "你好"}],
        max_tokens=10
    )
    print("✅ API 调用成功")
    print(f"回复: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()