"""
PDF 解析器 - 使用 MinerU（增强版：支持表格和图片）
"""
import subprocess
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PDFParser:
    """使用 MinerU 解析 PDF 为 Markdown（支持表格和图片）"""
    
    def __init__(
        self, 
        output_dir: str = "./data/processed",
        extract_images: bool = True,
        parse_method: str = "auto"  # auto, ocr, txt
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extract_images = extract_images
        self.parse_method = parse_method
    
    def parse(
        self, 
        pdf_path: str, 
        timeout: int = 300,
        lang: str = "ch"  # 支持语言指定
    ) -> Optional[Dict[str, str]]:
        """
        解析单个 PDF
        
        Args:
            pdf_path: PDF 文件路径
            timeout: 超时时间（秒）
            lang: 语言代码（ch=中文, en=英文）
            
        Returns:
            {
                'markdown': Markdown文件路径,
                'images': 图片目录路径,
                'content_json': 内容JSON路径
            }
        """
        try:
            # MinerU 命令（新版命令名是 mineru）
            cmd = [
                "mineru",  # 或 "magic-pdf" (旧版)
                "-p", pdf_path,
                "-o", str(self.output_dir),
                "-m", self.parse_method,  # auto会自动识别表格和图片
                "--device", "cpu"  # 强制使用 CPU，避免 GPU 显存不足
            ]
            
            # 添加语言参数（提升中文识别率）
            if lang:
                cmd.extend(["--lang", lang])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                # MinerU 输出结构:
                # output_dir/
                #   └── pdf_name/
                #       └── auto/                # MinerU 会创建这个子目录
                #           ├── pdf_name.md
                #           ├── pdf_name_content_list.json
                #           ├── pdf_name_model.json
                #           └── images/
                
                pdf_name = Path(pdf_path).stem
                result_base = self.output_dir / pdf_name
                
                # 递归查找 Markdown 文件（适配 MinerU 的实际输出结构）
                # 使用 *.md 而不是精确匹配，因为文件名可能被截断
                md_files = list(result_base.rglob("*.md"))
                
                if md_files:
                    md_path = md_files[0]
                    result_dir = md_path.parent
                    images_dir = result_dir / "images"
                    content_json = result_dir / f"{pdf_name}_content_list.json"
                    
                    logger.info(f"✅ 解析成功: {pdf_path}")
                    logger.info(f"   📄 Markdown: {md_path}")
                    
                    if images_dir.exists():
                        image_count = len(list(images_dir.glob("*")))
                        logger.info(f"   🖼️  提取图片: {image_count} 张")
                    
                    return {
                        'markdown': str(md_path),
                        'images': str(images_dir) if images_dir.exists() else None,
                        'content_json': str(content_json) if content_json.exists() else None
                    }
                else:
                    logger.warning(f"⚠️  解析失败: {pdf_path}")
                    logger.warning(f"   未找到输出文件，可能PDF内容为空或格式不支持")
                    return None
                    
            logger.warning(f"⚠️  解析失败: {pdf_path}")
            logger.warning(f"   错误信息: {result.stderr[:200]}")
            return None
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ 超时: {pdf_path}")
            return None
        except Exception as e:
            logger.error(f"❌ 异常: {pdf_path} - {e}")
            return None
    
    def batch_parse(self, pdf_dir: str, skip_existing: bool = True) -> list[Dict[str, str]]:
        """
        批量解析，返回所有成功解析的结果
        
        Args:
            pdf_dir: PDF 文件目录
            skip_existing: 是否跳过已解析的文件（默认 True）
        """
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        logger.info(f"📚 找到 {len(pdf_files)} 个PDF文件")
        
        parsed_results = []
        skipped_count = 0
        
        for pdf_path in pdf_files:
            pdf_name = pdf_path.stem
            
            # 检查是否已经解析过
            if skip_existing:
                # 检查输出目录中是否存在对应的 markdown 文件
                expected_md = self.output_dir / pdf_name / "auto" / f"{pdf_name}.md"
                if expected_md.exists():
                    logger.info(f"⏭️  跳过已解析: {pdf_path.name}")
                    # 返回已存在的结果
                    result_dir = expected_md.parent
                    images_dir = result_dir / "images"
                    content_json = result_dir / f"{pdf_name}_content_list.json"
                    
                    parsed_results.append({
                        'markdown': str(expected_md),
                        'images': str(images_dir) if images_dir.exists() else None,
                        'content_json': str(content_json) if content_json.exists() else None
                    })
                    skipped_count += 1
                    continue
            
            # 解析新文件
            result = self.parse(str(pdf_path))
            if result:
                parsed_results.append(result)
        
        logger.info(f"\n✅ 处理完成:")
        logger.info(f"   - 跳过已解析: {skipped_count} 个")
        logger.info(f"   - 新解析: {len(parsed_results) - skipped_count} 个")
        logger.info(f"   - 总计: {len(parsed_results)}/{len(pdf_files)} 个")
        
        total_images = sum(
            len(list(Path(r['images']).glob("*"))) 
            for r in parsed_results 
            if r.get('images') and Path(r['images']).exists()
        )
        logger.info(f"   - 提取图片: {total_images} 张")
        
        return parsed_results