"""
Milvus 客户端 - 向量存储与检索（使用硅基流动 Embedding API）
"""
from pymilvus import MilvusClient as PyMilvusClient
from openai import OpenAI
from typing import List, Dict
import logging
import os
import time

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus Lite 客户端（使用硅基流动 Embedding API）"""
    
    def __init__(
        self,
        db_path: str = "./data/milvus_lite.db",
        collection_name: str = "financial_reports",
        embedding_model: str = "BAAI/bge-m3"
    ):
        self.client = PyMilvusClient(db_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # 使用硅基流动 Embedding API
        logger.info(f"🔧 使用硅基流动 Embedding API: {embedding_model}")
        self.embedding_client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1"
        )
        logger.info("✅ Embedding API 初始化完成")
        
        self._create_collection()
    
    def _create_collection(self):
        """创建集合"""
        if self.client.has_collection(self.collection_name):
            logger.info(f"集合 {self.collection_name} 已存在")
            return
        
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=1024,  # BGE-M3
            metric_type="IP",  # 内积
        )
        logger.info(f"✅ 创建集合: {self.collection_name}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        批量向量化（调用硅基流动 API）
        
        Args:
            texts: 待向量化的文本列表
            batch_size: API 批量限制（SiliconFlow 限制为 64）
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"   📦 批次 {batch_num}/{total_batches}: 处理 {len(batch)} 个文本")
            
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
    
    def insert(self, chunks: List[Dict]):
        """
        插入文档块
        
        Args:
            chunks: [{'text': str, 'metadata': dict, 'chunk_id': int}, ...]
        """
        if not chunks:
            return
        
        texts = [c['text'] for c in chunks]
        total_batches = (len(texts) + 63) // 64  # 向上取整
        logger.info(f"🔧 正在向量化 {len(texts)} 个文本块（分 {total_batches} 批）...")
        vectors = self.embed_texts(texts)
        
        entities = []
        # 使用时间戳 + 索引生成唯一整数 ID
        base_id = int(time.time() * 1000)  # 毫秒级时间戳
        
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            entities.append({
                "id": base_id + idx,  # 整数 ID
                "vector": vec,
                "text": chunk['text'],
                "source": chunk['metadata'].get('source', ''),
                "chunk_id": chunk['chunk_id']
            })
        
        self.client.insert(
            collection_name=self.collection_name,
            data=entities
        )
        logger.info(f"✅ 插入 {len(entities)} 条记录")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """向量检索"""
        query_vec = self.embed_texts([query])[0]
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vec],
            limit=top_k,
            output_fields=["text", "source", "chunk_id"]
        )
            
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                'text': hit['entity']['text'],
                'source': hit['entity']['source'],
                'score': hit['distance']
            })
        return formatted_results
