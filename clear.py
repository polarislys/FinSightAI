"""
清空 processed 文件夹，准备重新解析
"""
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_processed_dir(processed_dir: str = "./data/processed"):
    """清空 processed 目录"""
    processed_path = Path(processed_dir)
    
    if not processed_path.exists():
        logger.info(f"✅ 目录不存在，无需清空: {processed_dir}")
        return
    
    # 删除所有子目录和文件
    for item in processed_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
            logger.info(f"🗑️  删除目录: {item.name}")
        else:
            item.unlink()
            logger.info(f"🗑️  删除文件: {item.name}")
    
    logger.info(f"✅ 已清空: {processed_dir}")


if __name__ == "__main__":
    clear_processed_dir()