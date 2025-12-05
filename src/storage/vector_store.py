"""
向量存储管理器 - 负责文本向量化和存储到 Milvus
"""
from typing import List, Dict
import logging

from src.retrieval.milvus_client import MilvusClient

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储管理器"""
    
    def __init__(
        self,
        db_path: str = "./data/milvus_lite.db",
        collection_name: str = "financial_reports",
        embedding_model: str = "BAAI/bge-m3"
    ):
        """
        初始化向量存储
        
        Args:
            db_path: Milvus 数据库路径
            collection_name: 集合名称
            embedding_model: Embedding 模型名称
        """
        self.milvus_client = MilvusClient(
            db_path=db_path,
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        logger.info(f"✅ VectorStore 初始化完成")
    
    def store_chunks(self, chunks: List[Dict]) -> bool:
        """
        向量化并存储文本块到 Milvus
        
        Args:
            chunks: 文本块列表
                [{'text': str, 'metadata': dict, 'chunk_id': int}, ...]
        
        Returns:
            是否成功
        """
        logger.info(f"\n{'='*60}")
        logger.info("🔢 向量化与存储")
        logger.info(f"{'='*60}")
        
        if not chunks:
            logger.warning("⚠️  没有数据需要向量化")
            return False
        
        try:
            logger.info(f"📊 准备向量化 {len(chunks)} 个文本块")
            
            # 调用 Milvus 客户端进行向量化和存储
            self.milvus_client.insert(chunks)
            
            logger.info(f"\n{'='*60}")
            logger.info("✅ 向量化并存储完成")
            logger.info(f"{'='*60}")
            logger.info(f"   - 存储切片: {len(chunks)} 个")
            logger.info(f"   - 数据库: {self.milvus_client.collection_name}")
            logger.info(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 向量化存储失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        向量检索
        
        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
        
        Returns:
            检索结果列表
        """
        return self.milvus_client.search(query, top_k=top_k)
    
    def get_stats(self) -> Dict:
        """
        获取存储统计信息
        
        Returns:
            统计信息字典
        """
        try:
            # 这里可以添加获取 Milvus 集合统计的代码
            # 目前返回基本信息
            return {
                'collection_name': self.milvus_client.collection_name,
                'embedding_model': self.milvus_client.embedding_model
            }
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return {}
