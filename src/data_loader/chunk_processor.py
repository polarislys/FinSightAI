"""
文本切分处理器 - 负责将解析后的文档切分成文本块
"""
from pathlib import Path
from typing import List, Dict
import logging

from src.data_loader.text_splitter import FinancialTextSplitter

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """文本切分处理器"""
    
    def __init__(self, chunk_size: int = 512):
        """
        初始化切分处理器
        
        Args:
            chunk_size: 每个文本块的最大 token 数
        """
        self.text_splitter = FinancialTextSplitter(chunk_size=chunk_size)
        logger.info(f"✅ ChunkProcessor 初始化完成 (chunk_size={chunk_size})")
    
    def process_parsed_results(self, parsed_results: List[Dict]) -> List[Dict]:
        """
        处理解析结果，切分成文本块
        
        Args:
            parsed_results: PDF 解析结果列表
                [{'markdown': 'path.md', 'images': 'path/images', ...}, ...]
        
        Returns:
            文本块列表
                [{'text': str, 'metadata': dict, 'chunk_id': int}, ...]
        """
        logger.info(f"\n{'='*60}")
        logger.info("✂️  文本切分处理")
        logger.info(f"{'='*60}")
        
        if not parsed_results:
            logger.warning("⚠️  没有解析结果需要处理")
            return []
        
        all_chunks = []
        
        for idx, result in enumerate(parsed_results, 1):
            md_path = result['markdown']
            source_name = Path(md_path).name
            
            # 读取 Markdown 文件
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                logger.error(f"❌ 读取文件失败: {md_path} - {e}")
                continue
            
            # 切分文本
            chunks = self.text_splitter.split_text(text)
            
            # 记录切分信息
            logger.info(f"\n📄 [{idx}/{len(parsed_results)}] {source_name}")
            logger.info(f"   - 原始文本长度: {len(text):,} 字符")
            logger.info(f"   - 切分后块数: {len(chunks)} 个")
            
            # 显示前 2 个块的预览
            for i, chunk in enumerate(chunks[:2]):
                preview = chunk[:100].replace('\n', ' ')
                logger.info(f"   - Chunk {i}: {preview}...")
            
            if len(chunks) > 2:
                logger.info(f"   - ... 还有 {len(chunks) - 2} 个块")
            
            # 添加到总列表
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'metadata': {
                        'source': source_name,
                        'doc_index': idx - 1
                    },
                    'chunk_id': i
                })
        
        # 统计信息
        logger.info(f"\n{'='*60}")
        logger.info("✅ 文本切分完成")
        logger.info(f"{'='*60}")
        logger.info(f"   - 处理文档: {len(parsed_results)} 个")
        logger.info(f"   - 总切片数: {len(all_chunks)} 个")
        if parsed_results:
            logger.info(f"   - 平均每文档: {len(all_chunks) / len(parsed_results):.1f} 个切片")
        logger.info(f"{'='*60}\n")
        
        return all_chunks
    
    def save_chunks_to_file(self, chunks: List[Dict], output_path: str):
        """
        将切片保存到文件（可选功能，用于调试）
        
        Args:
            chunks: 文本块列表
            output_path: 输出文件路径
        """
        import json
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 切片已保存到: {output_path}")
        except Exception as e:
            logger.error(f"❌ 保存切片失败: {e}")
