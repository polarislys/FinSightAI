"""
RAG 服务层 - 封装检索与问答核心逻辑，供 FastAPI 调用
"""
import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pymilvus import MilvusClient as PyMilvusClient
from dotenv import load_dotenv

from src.retrieval.milvus_client import MilvusClient
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker

load_dotenv()
logger = logging.getLogger(__name__)


class RAGService:
    """RAG 服务：检索 + 问答"""
    
    def __init__(
        self,
        db_path: str = "./data/milvus_lite.db",
        collection_name: str = "financial_reports"
    ):
        logger.info("🚀 初始化 RAGService...")
        
        # 1. 初始化 Milvus 向量检索
        self.milvus = MilvusClient(db_path, collection_name)
        
        # 2. 从 Milvus 加载文档，构建 BM25 索引
        documents = self._load_documents_from_milvus(collection_name)
        self.bm25 = BM25Retriever()
        self.bm25.add_documents(documents)
        
        # 3. 初始化 Reranker
        self.reranker = Reranker()
        
        # 4. 初始化混合检索器
        self.hybrid = HybridSearcher(self.milvus, self.bm25, self.reranker)
        
        # 5. 初始化 LLM 客户端
        self.llm = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1"
        )
        self.llm_model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        
        logger.info("✅ RAGService 初始化完成")
    
    def _load_documents_from_milvus(
        self, 
        collection_name: str,
        limit: int = 3000
    ) -> List[Dict]:
        """从 Milvus 加载文档用于 BM25 索引"""
        client = PyMilvusClient("./data/milvus_lite.db")
        results = client.query(
            collection_name=collection_name,
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
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        四路检索对比（用于调试）
        
        Returns:
            {
                "query": str,
                "top_k": int,
                "vector": [...],
                "bm25": [...],
                "rrf": [...],
                "rrf_rerank": [...]
            }
        """
        logger.info(f"🔍 检索: {query}")
        
        # 向量检索
        vector_results = self.milvus.search(query, top_k=top_k)
        
        # BM25 检索
        bm25_results = self.bm25.search(query, top_k=top_k)
        
        # RRF 融合（不 rerank）
        rrf_results = self.hybrid.search(query, top_k=top_k, use_rerank=False)
        
        # RRF 融合 + Rerank
        rrf_rerank_results = self.hybrid.search(query, top_k=top_k, use_rerank=True)
        
        return {
            "query": query,
            "top_k": top_k,
            "vector": self._format_results(vector_results),
            "bm25": self._format_results(bm25_results),
            "rrf": self._format_results(rrf_results),
            "rrf_rerank": self._format_results(rrf_rerank_results)
        }
    
    def _format_results(self, results: List[Dict]) -> List[Dict]:
        """统一结果格式"""
        formatted = []
        for r in results:
            source = r.get("source") or r.get("metadata", {}).get("source", "")
            # 清洗 em 标签
            source = re.sub(r'em', '', source)
            
            formatted.append({
                "text": r.get("text", "")[:500],
                "source": source,
                "score": r.get("score") or r.get("rrf_score") or r.get("rerank_score", 0)
            })
        return formatted
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        问答接口
        
        Args:
            messages: OpenAI 格式的消息列表 [{"role": "user", "content": "..."}]
            top_k: 检索文档数量
        
        Returns:
            {
                "answer": str,
                "source_documents": [...]
            }
        """
        # 1. 提取用户最后一条消息作为 query
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break
        
        if not query:
            return {
                "answer": "请提供您的问题。",
                "source_documents": []
            }
        
        logger.info(f"💬 问答: {query}")
        
        # 2. 检索（使用 RRF + Rerank）
        docs = self.hybrid.search(query, top_k=top_k, use_rerank=True)
        
        # 3. 构建上下文
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source") or doc.get("metadata", {}).get("source", "未知来源")
            text = doc.get("text", "")
            context_parts.append(f"[文档{i}] 来源: {source}\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # 4. 构建 prompt
        system_prompt = """你是一位专业的金融行业分析师。请根据提供的上下文片段回答用户问题。
        ## 回答要求：
        1. **综合叙述**：将同一家公司的信息整合在一起，不要机械罗列每条文档
        2. **穷尽实体**：提取上下文中**所有**被提及的相关公司，包括：
        - 公告发布方（主角）
        - 文中提到的竞争对手、合作伙伴、子公司（配角）
        3. **抓大放小**：如果未提及金额或日期，直接省略，不要写"未提及"
        4. **主次分明**：重点关注文档的"主角"公司；如果是A公司提到的竞品，请明确说明"A公司在报告中分析了..."
        5. **口语化专业**：使用流畅的中文，类似研报摘要风格
        6. **严格基于上下文**：只使用提供的文档内容，不要使用训练知识
        ## 输出格式：
        根据现有公告分析，[主题]的主要动态如下：
        **1. [公司名称] ([核心动作])**
        [2-3句话描述关键信息]
        **2. [公司名称] ([核心动作])**
        ...
        如上下文不足，直接回答"现有资料不足以回答"。"""
        user_prompt = f"""## 检索到的公告原文：
{context}

用户问题：{query}

请按照规范格式回答："""

        # 5. 调用 LLM
        try:
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ LLM 调用失败: {e}")
            answer = f"抱歉，生成回答时出现错误：{str(e)}"
        
        # 6. 返回结果
        source_documents = []
        for doc in docs:
            source_documents.append({
                "source": doc.get("source") or doc.get("metadata", {}).get("source", ""),
                "text": doc.get("text", "")[:300],
                "score": doc.get("rerank_score") or doc.get("rrf_score") or doc.get("score", 0)
            })
        
        return {
            "answer": answer,
            "source_documents": source_documents
        }