"""
混合检索 - 融合向量检索和 BM25 检索，使用 RRF 算法
"""
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class HybridSearcher:
    """混合检索器（向量 + BM25 + RRF 融合）"""
    
    def __init__(self, vector_store, bm25_retriever, reranker=None):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        logger.info("✅ HybridSearcher 初始化完成")
    
    def deduplicate_by_source(
        self, 
        results: List[Dict], 
        max_per_source: int = 2
    ) -> List[Dict]:
        """
        同源去重：每个来源（公司）最多保留 N 条
        
        Args:
            results: 检索结果列表
            max_per_source: 每个来源最多保留的条数
        
        Returns:
            去重后的结果列表
        """
        source_count = {}
        deduplicated = []
        
        for doc in results:
            source = doc.get("source") or doc.get("metadata", {}).get("source", "")
            # 提取公司名（假设格式是 "公司名_公告标题"）
            company = source.split("_")[0] if "_" in source else source
            
            if source_count.get(company, 0) < max_per_source:
                deduplicated.append(doc)
                source_count[company] = source_count.get(company, 0) + 1
        
        return deduplicated
    
    def rrf_fusion(
        self, 
        vector_results: List[Dict], 
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """RRF 融合算法"""
        rrf_scores = {}
        text_to_doc = {}
        
        for rank, doc in enumerate(vector_results, start=1):
            text = doc['text']
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (k + rank)
            if text not in text_to_doc:
                text_to_doc[text] = doc
        
        for doc in bm25_results:
            text = doc['text']
            rank = doc.get('rank', 1)
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (k + rank)
            if text not in text_to_doc:
                text_to_doc[text] = doc
        
        sorted_texts = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
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
        use_rerank: bool = True,
        use_diversity: bool = True,  # 新增：是否启用多样性去重
        max_per_source: int = 2,     # 新增：每个来源最多保留条数
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60
    ) -> List[Dict]:
        """混合检索"""
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
        
        # 4. Reranker 精排
        if use_rerank and self.reranker:
            logger.info("   🔄 执行 Reranker 精排...")
            fused_results = self.reranker.rerank(query, fused_results[:top_k * 2], top_k * 2)
        
        # 5. 同源去重（新增）
        if use_diversity:
            logger.info(f"   🎯 执行同源去重（每来源最多 {max_per_source} 条）...")
            fused_results = self.deduplicate_by_source(fused_results, max_per_source)
        
        final_results = fused_results[:top_k]
        logger.info(f"   ✅ 融合完成，返回 Top-{len(final_results)} 结果")
        return final_results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'vector_store': self.vector_store.get_stats(),
            'bm25_retriever': self.bm25_retriever.get_stats()
        }