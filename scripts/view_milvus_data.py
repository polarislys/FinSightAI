import os
from dotenv import load_dotenv
from pymilvus import MilvusClient
import json

load_dotenv()

def view_milvus_data():
    """查看 Milvus 数据库中的所有数据"""
    
    # 连接 Milvus
    try:
        client = MilvusClient(uri="http://localhost:19530")
        print("✅ 连接到 Docker Milvus 成功")
    except:
        try:
            client = MilvusClient(uri="./financial_rag.db")
            print("✅ 连接到 Milvus Lite")
        except Exception as e:
            print(f"❌ 无法连接 Milvus: {e}")
            return
    
    # 列出所有集合
    collections = client.list_collections()
    print(f"\n📚 数据库中的集合: {collections}")
    
    if not collections:
        print("❌ 数据库中没有集合")
        return
    
    # 遍历每个集合
    for collection_name in collections:
        print(f"\n{'='*60}")
        print(f"📦 集合名称: {collection_name}")
        print(f"{'='*60}")
        
        try:
            # 查询所有文档
            results = client.query(
                collection_name=collection_name,
                filter="id >= 0",
                output_fields=["id", "text", "metadata"],
                limit=100
            )
            
            print(f"📊 文档数量: {len(results)}")
            
            if not results:
                print("⚠️ 集合为空")
                continue
            
            # 显示每个文档
            for i, doc in enumerate(results, 1):
                print(f"\n--- 文档 {i} ---")
                print(f"ID: {doc.get('id', 'N/A')}")
                
                text = doc.get('text', '')
                if len(text) > 200:
                    print(f"文本: {text[:200]}...")
                else:
                    print(f"文本: {text}")
                
                metadata_str = doc.get('metadata', '{}')
                try:
                    metadata = json.loads(metadata_str)
                    print(f"元数据: {metadata}")
                except:
                    print(f"元数据: {metadata_str}")
                
                print("-" * 60)
        
        except Exception as e:
            print(f"❌ 查询集合 {collection_name} 失败: {e}")
    
    print(f"\n{'='*60}")
    print("✅ 数据查看完成")
    print(f"{'='*60}")

def delete_collection(collection_name):
    """删除指定集合"""
    try:
        client = MilvusClient(uri="http://localhost:19530")
    except:
        client = MilvusClient(uri="./financial_rag.db")
    
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"✅ 已删除集合: {collection_name}")
    else:
        print(f"⚠️ 集合不存在: {collection_name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "delete":
        if len(sys.argv) > 2:
            delete_collection(sys.argv[2])
        else:
            print("用法: python view_milvus_data.py delete <collection_name>")
    else:
        view_milvus_data()