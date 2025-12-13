"""
Reranker - 使用 BGE-Reranker-v2-m3 进行精排
"""
from typing import List, Dict
import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)


class Reranker:
    """使用硅基流动 Reranker API 进行精排"""
    
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3"):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1"
        )
        logger.info(f"✅ Reranker 初始化完成: {model}")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待排序文档列表 [{'text': str, ...}, ...]
            top_k: 返回前 k 个结果
        
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        logger.info(f"🔄 Reranker 精排 {len(documents)} 个文档...")
        
        # 提取文本
        texts = [doc['text'] for doc in documents]
        
        try:
            # 调用硅基流动 Reranker API
            response = self.client.post(
                "/rerank",
                body={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                    "top_n": min(top_k, len(texts)),
                    "return_documents": False
                },
                cast_to=object
            )
            
            # 解析结果
            results = response.get("results", [])
            
            # 按 rerank 分数重排
            reranked_docs = []
            for item in results:
                idx = item["index"]
                score = item["relevance_score"]
                doc = documents[idx].copy()
                doc["rerank_score"] = score
                reranked_docs.append(doc)
            
            # 按分数降序排序
            reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            logger.info(f"✅ Reranker 完成，返回 Top-{top_k}")
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Reranker 失败: {e}")
            # 降级：返回原始顺序
            return documents[:top_k]