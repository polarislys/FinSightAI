import os
from dotenv import load_dotenv
from pymilvus import MilvusClient
import json

load_dotenv()

def view_milvus_data(limit: int = None):
    """查看 Milvus 数据库中的所有数据"""
    
    # 连接 Milvus
    try:
        client = MilvusClient(uri="http://localhost:19530")
        print("✅ 连接到 Docker Milvus 成功")
    except:
        try:
            client = MilvusClient(uri="./data/milvus_lite.db")
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
            # 先获取总数
            stats = client.get_collection_stats(collection_name)
            total_count = stats.get('row_count', 0)
            print(f"📊 总文档数量: {total_count}")
            
            # 查询文档（如果没指定 limit，则查询全部，但最多显示前 20 条详情）
            query_limit = limit if limit else total_count
            results = client.query(
                collection_name=collection_name,
                filter="id >= 0",
                output_fields=["id", "text", "source", "chunk_id"],  # 使用正确的字段名
                limit=query_limit
            )
            
            print(f"📄 查询到: {len(results)} 条")
            
            if not results:
                print("⚠️ 集合为空")
                continue
            
            # 显示前 20 条详情
            display_count = min(20, len(results))
            print(f"\n--- 显示前 {display_count} 条文档 ---")
            
            for i, doc in enumerate(results[:display_count], 1):
                print(f"\n--- 文档 {i} ---")
                print(f"ID: {doc.get('id', 'N/A')}")
                print(f"Chunk ID: {doc.get('chunk_id', 'N/A')}")
                print(f"来源: {doc.get('source', 'N/A')}")
                
                text = doc.get('text', '')
                if len(text) > 200:
                    print(f"文本: {text[:200]}...")
                else:
                    print(f"文本: {text}")
                
                print("-" * 60)
            
            if len(results) > display_count:
                print(f"\n... 还有 {len(results) - display_count} 条文档未显示")
        
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
        client = MilvusClient(uri="./data/milvus_lite.db")
    
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
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        view_milvus_data(limit=int(sys.argv[1]))
    else:
        view_milvus_data()