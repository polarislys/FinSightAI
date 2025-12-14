"""
Sprint 1: 最小化 RAG 闭环
上传文档 -> 存入库 -> 提问 -> 回答
"""
import sys
sys.path.append('.')

from src.data_loader.pdf_parser_api import PDFParserAPI  # 改用 API 版本
from src.data_loader.text_splitter import FinancialTextSplitter
from src.retrieval.milvus_client import MilvusClient
from pathlib import Path
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
        self.pdf_parser = PDFParserAPI("./data/processed")  # 使用 API 解析器
        self.text_splitter = FinancialTextSplitter(chunk_size=512)
        self.milvus = MilvusClient("./data/milvus_lite.db")
        
        # SiliconFlow (Qwen 7B)
        self.llm = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1"
        )
        logger.info("✅ 初始化完成")
    
    def ingest_pdfs(self, pdf_dir: str):
        """数据摄取：PDF解析 + 切分 + 入库"""
        logger.info(f"\n{'='*60}")
        logger.info("� Step 1: PDF 解析（MinerU）")
        logger.info(f"{'='*60}")
        
        # 1. 解析PDF
        parsed_results = self.pdf_parser.batch_parse(pdf_dir)
        
        if not parsed_results:
            logger.error("❌ 没有成功解析的PDF")
            return
        
        # 2. 读取Markdown并切分
        logger.info(f"\n{'='*60}")
        logger.info("✂️  Step 2: 文本切分（TokenTextSplitter）")
        logger.info(f"{'='*60}")
        
        all_chunks = []
        for result in parsed_results:
            md_path = result['markdown']
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 从 md_path 提取 PDF 原始文件名（目录名就是 PDF 文件名）
            # 例如: data/processed/豫能控股_关于投资建设.../full.md
            # 目录名: 豫能控股_关于投资建设...
            pdf_name = Path(md_path).parent.name
            
            chunks = self.text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'metadata': {'source': pdf_name},  # 使用 PDF 文件名
                    'chunk_id': i
                })
        
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
    main()