"""
混合检索 - 融合向量检索和 BM25 检索，使用 RRF 算法
"""
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class HybridSearcher:
    """混合检索器（向量 + BM25 + RRF 融合）"""
    
    def __init__(self, vector_store, bm25_retriever):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储实例
            bm25_retriever: BM25 检索器实例
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker  # 新增
        logger.info("✅ HybridSearcher 初始化完成")
    
    def rrf_fusion(
        self, 
        vector_results: List[Dict], 
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法
        
        公式: RRF_score = Σ 1/(k + rank_i)
        
        Args:
            vector_results: 向量检索结果 [{'text': str, 'score': float, ...}, ...]
            bm25_results: BM25 检索结果 [{'text': str, 'score': float, 'rank': int}, ...]
            k: RRF 参数，通常取 60
        
        Returns:
            融合后的结果列表（按 RRF 分数降序）
        """
        # 构建文本到分数的映射
        rrf_scores = {}
        text_to_doc = {}
        
        # 处理向量检索结果
        for rank, doc in enumerate(vector_results, start=1):
            text = doc['text']
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (k + rank)
            if text not in text_to_doc:
                text_to_doc[text] = doc
        
        # 处理 BM25 检索结果
        for doc in bm25_results:
            text = doc['text']
            rank = doc.get('rank', 1)
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (k + rank)
            if text not in text_to_doc:
                text_to_doc[text] = doc
        
        # 按 RRF 分数排序
        sorted_texts = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 构建最终结果
        fused_results = []
        for text, rrf_score in sorted_texts:
            doc = text_to_doc[text].copy()
            doc['rrf_score'] = rrf_score
            fused_results.append(doc)
        
        return fused_results
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        use_rerank: bool = True,  # 新增
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60
    ) -> List[Dict]:
        """
        混合检索（向量 + BM25 + RRF 融合）
        
        Args:
            query: 查询文本
            top_k: 最终返回的结果数量
            vector_weight: 向量检索的权重（暂未使用，RRF 自动平衡）
            bm25_weight: BM25 检索的权重（暂未使用，RRF 自动平衡）
            rrf_k: RRF 参数
        
        Returns:
            融合后的检索结果
        """
        logger.info(f"🔍 混合检索: {query}")
        
        # 1. 向量检索
        logger.info("   📊 执行向量检索...")
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        logger.info(f"   ✅ 向量检索返回 {len(vector_results)} 个结果")
        
        # 2. BM25 检索
        logger.info("   📝 执行 BM25 检索...")
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
        logger.info(f"   ✅ BM25 检索返回 {len(bm25_results)} 个结果")
        
        # 3. RRF 融合
        logger.info("   🔀 执行 RRF 融合...")
        fused_results = self.rrf_fusion(vector_results, bm25_results, k=rrf_k)
        # 新增：Reranker 精排
        if use_rerank and self.reranker:
            logger.info("   🔄 执行 Reranker 精排...")
            final_results = self.reranker.rerank(query, fused_results[:top_k * 2], top_k)
        else:
            final_results = fused_results[:top_k]
        
        logger.info(f"   ✅ 融合完成，返回 Top-{top_k} 结果")
        return final_results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'vector_store': self.vector_store.get_stats(),
            'bm25_retriever': self.bm25_retriever.get_stats()
        }