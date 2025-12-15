"""
Sprint 1: 最小化 RAG 闭环
上传文档 -> 存入库 -> 提问 -> 回答
"""
import sys
sys.path.append('.')

from src.data_loader.pdf_parser_api import PDFParserAPI  # 改用 API 版本
from src.data_loader.text_splitter import FinancialTextSplitter
from src.data_loader.chunk_processor import ChunkProcessor
from src.retrieval.milvus_client import MilvusClient
from pathlib import Path
import os
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
# 设置 Hugging Face 缓存（在初始化模型之前）
os.environ['HF_HOME'] = os.getenv('HF_HOME', '/home/nl/disk_8T/lys/cache/huggingface')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Sprint1Pipeline:
    """第一阶段：基础 RAG 管道"""
    
    def __init__(self):
        self.text_splitter = FinancialTextSplitter(chunk_size=512)
        self.chunk_processor = ChunkProcessor(chunk_size=512)  # 🔥 新增
        self.milvus = MilvusClient("./data/milvus_lite.db")
        
        # SiliconFlow (Qwen 7B)
        self.llm = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1"
        )
        logger.info("✅ 初始化完成")
    
    def ingest_pdfs(self, pdf_dir: str):
        """数据摄取：PDF解析 + 切分 + 入库"""
        # 🔥 根据输入目录确定输出目录
        
        pdf_dir_path = Path(pdf_dir)
        subfolder = pdf_dir_path.name  # "announcements" 或 "research_reports"
        output_dir = f"./data/processed/{subfolder}"
    
        # 重新初始化 parser 使用正确的输出目录
        self.pdf_parser = PDFParserAPI(output_dir)
        logger.info(f"\n{'='*60}")
        logger.info("📄 Step 1: PDF 解析（MinerU）")
        logger.info(f"{'='*60}")
        
        # 1. 解析PDF (内置跳过逻辑)
        parsed_results = self.pdf_parser.batch_parse(pdf_dir, skip_existing=True)
        
        if not parsed_results:
            logger.error("❌ 没有成功解析的PDF")
            return
        
        # 2. 使用chunk_processor处理 (自动添加doc_type)
        all_chunks = self.chunk_processor.process_parsed_results(parsed_results)

        # 3. 向量化并入库
        logger.info(f"\n{'='*60}")
        logger.info("🔢 Step 3: 向量化入库（BGE-M3 + Milvus）")
        logger.info(f"{'='*60}")

        self.milvus.insert(all_chunks)

        logger.info(f"\n✅ 数据摄取完成！")
        logger.info(f"   - 解析PDF: {len(parsed_results)} 个")
        logger.info(f"   - 生成chunks: {len(all_chunks)} 个")
        
    
    def query(self, question: str) -> str:
        """朴素 RAG 查询"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 用户提问: {question}")
        logger.info(f"{'='*60}")
        
        # 1. 向量检索
        results = self.milvus.search(question, top_k=3)
        
        logger.info(f"\n📊 检索结果:")
        for i, res in enumerate(results, 1):
            logger.info(f"{i}. [相似度: {res['score']:.3f}] {res['source']}")
            logger.info(f"   {res['text'][:80]}...\n")
        
        # 2. 构建 Prompt
        context = "\n\n".join([r['text'] for r in results])
        prompt = f"""你是一个专业的金融分析助手。根据以下参考资料回答问题。

【参考资料】
{context}

【用户问题】
{question}

【要求】
1. 仅基于参考资料回答
2. 如果资料中没有相关信息，明确说明
3. 回答要简洁专业

【回答】"""
        
        # 3. 调用 Qwen 7B
        logger.info("💬 调用 SiliconFlow (Qwen 7B)...")
        response = self.llm.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512
        )
        
        answer = response.choices[0].message.content
        
        logger.info(f"\n🤖 AI 回答:\n{answer}\n")
        return answer


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sprint 1: 最小化RAG闭环')
    parser.add_argument('--ingest', action='store_true', help='数据摄取')
    parser.add_argument('--query', type=str, help='查询问题')
    args = parser.parse_args()
    
    pipeline = Sprint1Pipeline()
    
    if args.ingest:
        pipeline.ingest_pdfs("./data/raw_pdfs/research_reports")
    
    if args.query:
        pipeline.query(args.query)
    
    # 交互模式
    if not args.ingest and not args.query:
        print("\n💬 进入交互模式（输入 'quit' 退出）\n")
        while True:
            question = input("🙋 你的问题: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question:
                pipeline.query(question)


if __name__ == "__main__":
    pipeline = Sprint1Pipeline()
    
    # 🔥 修改：处理所有子文件夹
    import os
    base_dir = "./data/raw_pdfs"
    for subfolder in ["announcements", "research_reports"]:  # 只处理有数据的文件夹
        folder_path = os.path.join(base_dir, subfolder)
        if os.path.exists(folder_path) and os.listdir(folder_path):
            print(f"\n处理文件夹: {subfolder}")
            pipeline.ingest_pdfs(folder_path)