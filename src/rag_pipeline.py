import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

# 加载环境变量
load_dotenv()

class FinancialRAGSystem:
    def __init__(self):
        # 初始化 OpenAI 客户端（用于 LLM 和 Embedding）
        self.client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL")
        )
        
            # 初始化 Milvus 客户端 - 连接到 Docker
        try:
            # 连接到 Docker Milvus
            self.milvus_client = MilvusClient(uri="http://localhost:19530")
            print("✅ 连接到 Docker Milvus 成功")
        except Exception as e:
            print(f"❌ 连接 Docker Milvus 失败: {e}")
            # 备选：使用 Milvus Lite
            try:
                self.milvus_client = MilvusClient(uri="./financial_rag.db")
                print("✅ 使用 Milvus Lite 作为备选")
            except Exception as e2:
                print(f"❌ Milvus Lite 也失败: {e2}")
                raise
        
        self.collection_name = "financial_documents"
        
        # 文本切片器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        # 初始化集合
        self._init_collection()
    
    def _init_collection(self):
        """初始化 Milvus 集合"""
        try:
            if self.milvus_client.has_collection(self.collection_name):
                self.milvus_client.drop_collection(self.collection_name)
            
            # BGE-M3 的向量维度是 1024
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=1024,
                metric_type="COSINE"
            )
            print(f"✅ 已创建集合: {self.collection_name}")
        except Exception as e:
            print(f"❌ 创建集合失败: {e}")
            raise
    
    # ... 其余方法保持不变
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的 embedding 向量"""
        try:
            response = self.client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL"),
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding 失败: {e}")
            return None
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """添加文档到向量库"""
        try:
            # 文档切片
            chunks = self.text_splitter.split_text(text)
            print(f"📄 文档已切分为 {len(chunks)} 个片段")
            
            # 准备数据
            data_to_insert = []
            for i, chunk in enumerate(chunks):
                embedding = self.get_embedding(chunk)
                if embedding is None:
                    continue
                
                data_to_insert.append({
                    "id": abs(hash(chunk)) % (2**63 - 1),
                    "vector": embedding,
                    "text": chunk,
                    "metadata": json.dumps(metadata or {})
                })
            
            # 插入数据
            if data_to_insert:
                self.milvus_client.insert(
                    collection_name=self.collection_name,
                    data=data_to_insert
                )
                print(f"✅ 已插入 {len(data_to_insert)} 个文档片段")
                return True
            else:
                print("❌ 没有有效的文档片段可插入")
                return False
                
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索相关文档"""
        try:
            # 获取查询的 embedding
            query_embedding = self.get_embedding(query)
            if query_embedding is None:
                return []
            
            # 在 Milvus 中搜索
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text", "metadata"]
            )
            
            # 格式化结果
            formatted_results = []
            for result in results[0]:
                formatted_results.append({
                    "text": result["entity"]["text"],
                    "score": result["distance"],
                    "metadata": json.loads(result["entity"]["metadata"])
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def ask(self, question: str) -> str:
        """RAG 问答"""
        try:
            # 检索相关文档
            relevant_docs = self.search(question, top_k=3)
            
            if not relevant_docs:
                return "抱歉，我没有找到相关的文档信息来回答您的问题。"
            
            # 构建上下文
            context = "\n\n".join([doc["text"] for doc in relevant_docs])
            
            # 构建 prompt
            prompt = f"""基于以下上下文信息，请回答用户的问题。如果上下文中没有相关信息，请诚实地说不知道。

上下文信息：
{context}

用户问题：{question}

请提供准确、有用的回答："""

            # 调用 LLM
            response = self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=[
                    {"role": "system", "content": "你是一个专业的金融助手，请基于提供的上下文信息准确回答用户问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ 问答失败: {e}")
            return f"抱歉，处理您的问题时出现了错误：{str(e)}"

# 示例金融文档数据
SAMPLE_FINANCIAL_DOCS = [
    {
        "text": """
        股票投资基础知识：
        股票是公司所有权的证明，代表投资者对公司的部分所有权。投资股票的主要目的是获得资本增值和股息收入。
        
        股票投资的主要风险包括：
        1. 市场风险：整体市场下跌导致的损失
        2. 公司风险：特定公司经营不善导致的风险
        3. 流动性风险：无法及时买卖股票的风险
        
        投资策略建议：
        - 分散投资，不要把所有资金投入单一股票
        - 长期持有，避免频繁交易
        - 定期评估投资组合表现
        """,
        "metadata": {"source": "股票投资指南", "category": "投资基础"}
    }
]

def main():
    """主函数：演示 RAG 系统使用"""
    print("🚀 初始化金融 RAG 系统...")
    
    try:
        rag_system = FinancialRAGSystem()
        
        # 添加示例文档
        print("\n📚 添加示例金融文档...")
        for doc in SAMPLE_FINANCIAL_DOCS:
            rag_system.add_document(doc["text"], doc["metadata"])
        
        print("\n" + "="*50)
        print("✅ RAG 系统初始化完成！")
        print("="*50)
        
        # 测试问题
        test_question = "什么是股票？投资股票有什么风险？"
        print(f"\n🤖 测试问答...")
        print(f"❓ {test_question}")
        answer = rag_system.ask(test_question)
        print(f"💡 {answer}")
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")

if __name__ == "__main__":
    main()