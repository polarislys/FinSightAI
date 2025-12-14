# scripts/clean_em_tags.py
import os
import re
from pathlib import Path

def clean_em_tags(text: str) -> str:
    """清洗 em 标签及其变体"""
    # 1. 匹配 <em>xxx</em> 标签
    text = re.sub(r'</?em>', '', text)
    # 2. 匹配中文环境中的 em（如 em新能源em、em充电emem桩em）
    #    关键：em 前后可能是中文字符，不能用 \b
    text = re.sub(r'em(?=[\u4e00-\u9fff])', '', text)  # em后面跟中文
    text = re.sub(r'(?<=[\u4e00-\u9fff])em', '', text)  # em前面是中文
    # 3. 处理连续的 emem 情况
    text = re.sub(r'emem', '', text)
    # 4. 处理残留的独立 em（英文边界）
    text = re.sub(r'\bem\b', '', text)
    return text

def rename_files(directory: str):
    """重命名文件夹和文件，去除 em 标签"""
    # 先处理目录（从深到浅），再处理文件
    all_paths = list(Path(directory).rglob('*'))
    # 按路径深度降序排列（先处理深层目录）
    all_paths.sort(key=lambda p: len(p.parts), reverse=True)
    
    for path in all_paths:
        if 'em' in path.name:
            new_name = clean_em_tags(path.name)
            if new_name != path.name:  # 只有真正变化了才重命名
                new_path = path.parent / new_name
                if not new_path.exists():
                    print(f"✅ 重命名: {path.name} -> {new_name}")
                    path.rename(new_path)
                else:
                    print(f"⚠️ 跳过（目标已存在）: {path.name}")

def clean_markdown_content(directory: str):
    """清洗 markdown 文件内容"""
    for md_file in Path(directory).rglob('*.md'):
        content = md_file.read_text(encoding='utf-8')
        cleaned = clean_em_tags(content)
        if content != cleaned:
            md_file.write_text(cleaned, encoding='utf-8')
            print(f"✅ 已清洗内容: {md_file.name}")

if __name__ == "__main__":
    print("🔧 开始清洗 em 标签...")
    # 1. 重命名文件和目录
    rename_files("./data/processed")
    # 2. 清洗 markdown 内容
    clean_markdown_content("./data/processed")
    print("✅ 清洗完成，请重新运行 sprint1_pipeline.py 入库")