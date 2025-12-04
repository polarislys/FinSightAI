print("🔍 测试 Milvus Lite 安装...")

try:
    from pymilvus import MilvusClient
    print("✅ pymilvus 导入成功")
    
    # 测试 Lite 连接
    client = MilvusClient(uri="./test_lite.db")
    print("✅ Milvus Lite 连接成功")
    
    # 测试基本操作
    collections = client.list_collections()
    print(f"✅ 基本操作成功，集合列表: {collections}")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    print("请运行: pip install pymilvus[milvus_lite]")