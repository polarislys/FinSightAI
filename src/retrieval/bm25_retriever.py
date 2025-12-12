"""
BM25 稀疏检索器 - 基于关键词匹配的检索
"""
from typing import List, Dict
import logging
from rank_bm25 import BM25Okapi
import jieba

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 稀疏检索器（使用 jieba 分词）"""
    
    def __init__(self):
        self.corpus = []  # 存储原始文本
        self.tokenized_corpus = []  # 存储分词后的文本
        self.bm25 = None
        self.metadata_list = []  # 存储元数据
        logger.info("✅ BM25Retriever 初始化完成")
    
    def add_documents(self, documents: List[Dict]):
        """
        添加文档到 BM25 索引
        
        Args:
            documents: [{'text': str, 'metadata': dict}, ...]
        """
        if not documents:
            logger.warning("⚠️  没有文档需要索引")
            return
        
        logger.info(f"🔧 正在为 {len(documents)} 个文档建立 BM25 索引...")
        
        for doc in documents:
            text = doc['text']
            self.corpus.append(text)
            self.metadata_list.append(doc.get('metadata', {}))
            
            # 使用 jieba 分词
            tokens = list(jieba.cut(text))
            self.tokenized_corpus.append(tokens)
        
        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"✅ BM25 索引构建完成，共 {len(self.corpus)} 个文档")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        BM25 检索
        
        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
        
        Returns:
            [{'text': str, 'score': float, 'metadata': dict, 'rank': int}, ...]
        """
        if self.bm25 is None:
            logger.warning("⚠️  BM25 索引未初始化")
            return []
        
        # 查询分词
        query_tokens = list(jieba.cut(query))
        
        # BM25 打分
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取 top_k 结果
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                'text': self.corpus[idx],
                'score': float(scores[idx]),
                'metadata': self.metadata_list[idx],
                'rank': rank
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            'total_documents': len(self.corpus),
            'indexed': self.bm25 is not None
        }