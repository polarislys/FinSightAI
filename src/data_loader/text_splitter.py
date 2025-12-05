"""
文本切分器 - 使用 TokenTextSplitter（按 README 要求）
"""
from langchain_text_splitters import TokenTextSplitter
from typing import List
import logging

logger = logging.getLogger(__name__)


class FinancialTextSplitter:
    """金融文档文本切分器"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"  # GPT-3.5/GPT-4 的tokenizer
    ):
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name
        )
        logger.info(f"初始化 TokenTextSplitter: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def split_text(self, text: str) -> List[str]:
        """切分文本"""
        if not text or len(text.strip()) < 10:
            return []
        
        chunks = self.splitter.split_text(text)
        logger.debug(f"文本切分完成: {len(chunks)} 个 chunks")
        return chunks
    
    def split_documents(self, documents: List[dict]) -> List[dict]:
        """
        切分文档列表
        
        Args:
            documents: [{'text': str, 'metadata': dict}, ...]
            
        Returns:
            [{'text': str, 'metadata': dict, 'chunk_id': int}, ...]
        """
        results = []
        for doc in documents:
            chunks = self.split_text(doc['text'])
            for i, chunk in enumerate(chunks):
                results.append({
                    'text': chunk,
                    'metadata': doc.get('metadata', {}),
                    'chunk_id': i
                })
        
        logger.info(f"切分 {len(documents)} 个文档 -> {len(results)} 个 chunks")
        return results