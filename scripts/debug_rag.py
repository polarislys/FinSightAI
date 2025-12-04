import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class DebugRAGSystem:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL")
        )
        
        try:
            self.milvus_client = MilvusClient(uri="http://localhost:19530")
            print("✅ 连接到 Docker Milvus 成功")
        except:
            self.milvus_client = MilvusClient(uri="./debug_rag.db")
            print("✅ 使用 Milvus Lite")
        
        self.collection_name = "debug_docs"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        self._init_collection()
    
    def _init_collection(self):
        try:
            if self.milvus_client.has_collection(self.collection_name):
                self.milvus_client.drop_collection(self.collection_name)
            
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=1024,
                metric_type="COSINE"
            )
            print(f"✅ 已创建集合: {self.collection_name}")
        except Exception as e:
            print(f"❌ 创建集合失败: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL"),
                input=text
            )
            embedding = response.data[0].embedding
            print(f"🔍 获取 embedding 成功，维度: {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"❌ Embedding 失败: {e}")
            return None
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        try:
            chunks = self.text_splitter.split_text(text)
            print(f"📄 文档已切分为 {len(chunks)} 个片段")
            
            data_to_insert = []
            for i, chunk in enumerate(chunks):
                print(f"   处理片段 {i+1}: {chunk[:100]}...")
                embedding = self.get_embedding(chunk)
                if embedding is None:
                    continue
                
                data_to_insert.append({
                    "id": abs(hash(chunk)) % (2**63 - 1),
                    "vector": embedding,
                    "text": chunk,
                    "metadata": json.dumps(metadata or {})
                })
            
            if data_to_insert:
                self.milvus_client.insert(
                    collection_name=self.collection_name,
                    data=data_to_insert
                )
                print(f"✅ 已插入 {len(data_to_insert)} 个文档片段")
                # 等待一下让数据被索引
                import time
                time.sleep(1)
                
                # 验证插入
                try:
                    docs = self.milvus_client.query(
                        collection_name=self.collection_name,
                        filter="id >= 0",  # 简单的过滤条件
                        output_fields=["id"],
                        limit=1000
                    )
                    print(f"🔍 集合中现有文档数量: {len(docs)}")
                except Exception as e:
                    print(f"🔍 查询文档数量失败: {e}")
                return True
            else:
                print("❌ 没有有效的文档片段可插入")
                return False
                
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        try:
            print(f"\n🔍 搜索查询: '{query}'")
            query_embedding = self.get_embedding(query)
            if query_embedding is None:
                return []
            
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text", "metadata"]
            )
            
            print(f"🔍 搜索返回 {len(results[0])} 个结果")
            
            formatted_results = []
            for i, result in enumerate(results[0]):
                score = result["distance"]
                text = result["entity"]["text"]
                print(f"   结果 {i+1}: 相似度={score:.4f}, 文本='{text[:100]}...'")
                
                formatted_results.append({
                    "text": text,
                    "score": score,
                    "metadata": json.loads(result["entity"]["metadata"])
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def ask(self, question: str) -> str:
        try:
            relevant_docs = self.search(question, top_k=3)
            
            if not relevant_docs:
                return "抱歉，我没有找到相关的文档信息来回答您的问题。"
            
            # 显示使用的上下文
            context = "\n\n".join([doc["text"] for doc in relevant_docs])
            print(f"\n📖 使用的上下文:\n{context}\n")
            
            prompt = f"""基于以下上下文信息，请回答用户的问题。

上下文信息：
{context}

用户问题：{question}

请提供准确、有用的回答："""

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

def main():
    print("🚀 启动调试 RAG 系统...")
    
    try:
        rag = DebugRAGSystem()
        
        # 添加测试文档
        test_doc = """
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
        """
        
        metadata = {"source": "股票投资指南", "category": "投资基础"}
        
        print("\n📚 添加示例文档...")
        if rag.add_document(test_doc, metadata):
            print("\n" + "="*50)
            print("✅ 调试 RAG 系统初始化完成！")
            print("="*50)
            
            # 测试问题
            question = "什么是股票？投资股票有什么风险？"
            print(f"\n🤖 测试问答...")
            print(f"❓ {question}")
            answer = rag.ask(question)
            print(f"💡 {answer}")
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()