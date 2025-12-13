"""
测试混合检索功能
"""
import sys
sys.path.append('/home/nl/disk_8T/lys/FinSightAI')

from src.retrieval import BM25Retriever, HybridSearcher
from src.storage.vector_store import VectorStore

# 测试文档
test_docs = [
    {
        'text': '贵州茅台2024年第一季度营收达到350亿元，同比增长15%，净利润180亿元。',
        'metadata': {'source': '茅台财报', 'year': 2024}
    },
    {
        'text': '五粮液2024年营收预计突破800亿元，市场份额持续扩大。',
        'metadata': {'source': '五粮液公告', 'year': 2024}
    },
    {
        'text': '白酒行业整体呈现高端化趋势，茅台和五粮液占据市场主导地位。',
        'metadata': {'source': '行业分析', 'category': '白酒'}
    }
]

def main():
    print("="*60)
    print("🧪 测试混合检索功能")
    print("="*60)
    
    # 1. 初始化 BM25
    print("\n1️⃣ 初始化 BM25 检索器...")
    bm25 = BM25Retriever()
    bm25.add_documents(test_docs)
    
    # 2. 初始化向量存储
    print("\n2️⃣ 初始化向量存储...")
    vector_store = VectorStore()
    
    # 准备向量存储格式的数据
    chunks = []
    for i, doc in enumerate(test_docs):
        chunks.append({
            'text': doc['text'],
            'metadata': doc['metadata'],
            'chunk_id': i
        })
    vector_store.store_chunks(chunks)
    
    # 3. 初始化混合检索
    print("\n3️⃣ 初始化混合检索器...")
    hybrid = HybridSearcher(vector_store, bm25)
    
    # 4. 测试检索
    print("\n4️⃣ 测试混合检索...")
    query = "茅台的营收是多少？"
    print(f"\n查询: {query}")
    
    results = hybrid.search(query, top_k=3)
    
    print(f"\n📊 检索结果 (Top-{len(results)}):")
    for i, result in enumerate(results, 1):
        print(f"\n--- 结果 {i} ---")
        print(f"文本: {result['text'][:50]}...")
        print(f"RRF分数: {result.get('rrf_score', 0):.4f}")
        if 'score' in result:
            print(f"原始分数: {result['score']:.4f}")
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)

if __name__ == "__main__":
    main()