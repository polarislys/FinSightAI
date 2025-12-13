"""
Sprint 2 测试脚本：BM25 + RRF 混合检索
"""
import sys
sys.path.append('.')

from src.retrieval.milvus_client import MilvusClient
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_search import HybridSearcher
from pathlib import Path
from src.retrieval.reranker import Reranker
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents_from_milvus(milvus_client, limit=3000):
    """从 Milvus 加载文档用于 BM25 索引"""
    from pymilvus import MilvusClient as PyMilvusClient
    
    client = PyMilvusClient("./data/milvus_lite.db")
    results = client.query(
        collection_name="financial_reports",
        filter="id >= 0",
        output_fields=["text", "source"],
        limit=limit
    )
    
    documents = []
    for doc in results:
        documents.append({
            'text': doc.get('text', ''),
            'metadata': {'source': doc.get('source', '')}
        })
    
    logger.info(f"📚 从 Milvus 加载了 {len(documents)} 个文档")
    return documents


def test_bm25_only(bm25_retriever, query):
    """测试纯 BM25 检索"""
    print(f"\n{'='*60}")
    print(f"📝 BM25 检索测试")
    print(f"{'='*60}")
    print(f"查询: {query}\n")
    
    results = bm25_retriever.search(query, top_k=5)
    
    for i, res in enumerate(results, 1):
        print(f"--- 结果 {i} (BM25 分数: {res['score']:.4f}) ---")
        print(f"来源: {res['metadata'].get('source', 'N/A')}")
        print(f"文本: {res['text'][:150]}...")
        print()


def test_vector_only(milvus_client, query):
    """测试纯向量检索"""
    print(f"\n{'='*60}")
    print(f"🔢 向量检索测试")
    print(f"{'='*60}")
    print(f"查询: {query}\n")
    
    results = milvus_client.search(query, top_k=5)
    
    for i, res in enumerate(results, 1):
        print(f"--- 结果 {i} (相似度: {res['score']:.4f}) ---")
        print(f"来源: {res.get('source', 'N/A')}")
        print(f"文本: {res['text'][:150]}...")
        print()


def test_hybrid_with_rerank(hybrid_searcher, query):
    """测试混合检索 + Reranker"""
    print(f"\n{'='*60}")
    print(f"🎯 混合检索 + Reranker 测试")
    print(f"{'='*60}")
    print(f"查询: {query}\n")
    
    results = hybrid_searcher.search(query, top_k=5, use_rerank=True)
    
    for i, res in enumerate(results, 1):
        score = res.get('rerank_score', res.get('rrf_score', 0))
        print(f"--- 结果 {i} (Rerank 分数: {score:.4f}) ---")
        source = res.get('metadata', {}).get('source', res.get('source', 'N/A'))
        print(f"来源: {source}")
        print(f"文本: {res['text'][:150]}...")
        print()


def main():
    print("="*60)
    print("🚀 Sprint 2 测试：BM25 + RRF 混合检索")
    print("="*60)
    
    # 1. 初始化 Milvus
    logger.info("初始化 Milvus...")
    milvus_client = MilvusClient("./data/milvus_lite.db")
    
    # 2. 加载文档并构建 BM25 索引
    logger.info("加载文档并构建 BM25 索引...")
    documents = load_documents_from_milvus(milvus_client)
    
    bm25_retriever = BM25Retriever()
    bm25_retriever.add_documents(documents)
    
    reranker = Reranker()
    
    # 初始化混合检索器（带 Reranker）
    hybrid_searcher = HybridSearcher(milvus_client, bm25_retriever, reranker)
    
    # 4. 测试查询
    test_queries = [
        "新能源汽车投资机会",
        "动力电池产能扩张",
        "储能电站项目",
        "充电桩建设",
    ]
    

    for query in test_queries:
        # 对比四种检索方式
        test_vector_only(milvus_client, query)
        test_bm25_only(bm25_retriever, query)
        test_hybrid_search(hybrid_searcher, query)  # 不带 rerank
        test_hybrid_with_rerank(hybrid_searcher, query)  # 带 rerank
        
        input("\n按 Enter 继续下一个查询...")
    
    print("\n✅ Sprint 2 测试完成！")


if __name__ == "__main__":
    main()