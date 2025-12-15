"""
PDF 解析器 - 使用 MinerU 官方 API + cninfo URL 反查
"""
import requests
import time
import zipfile
import io
from pathlib import Path
from typing import Optional, Dict, List
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# cninfo 配置
CNINFO_API = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "http://www.cninfo.com.cn/new/disclosure",
}


class PDFParserAPI:
    """使用 MinerU 官方 API 解析 PDF"""
    
    def __init__(
        self, 
        output_dir: str = "./data/processed",
        api_token: str = None,
        user_token: str = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MinerU API Token
        self.api_token = api_token or os.getenv("MINERU_API_TOKEN")
        self.user_token = user_token or os.getenv("MINERU_USER_TOKEN", "default_user")
        
        if not self.api_token:
            raise ValueError("需要设置 MINERU_API_TOKEN 环境变量")
        
        self.api_base_url = "https://mineru.net/api/v4/extract"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
            "token": self.user_token
        }
        
        logger.info(f"✅ MinerU API 解析器初始化完成")
        logger.info(f"   API: {self.api_base_url}")
    
    def _resolve_cninfo_pdf_url(self, pdf_filename: str) -> Optional[str]:
        """
        从文件名反查 cninfo 获取 PDF URL
        
        文件名格式: 公司简称_公告标题.pdf
        例如: 中恒电气_关于与关联人共同投资设立合资公司...公告.pdf
        """
        try:
            stem = Path(pdf_filename).stem
            
            # 分离公司简称和公告标题
            if '_' in stem:
                sec_name, title = stem.split('_', 1)
            else:
                sec_name = ""
                title = stem
            
            # 清理标题中的 em 标记
            title_clean = title.replace("em", "")
            
            logger.info(f"   🔍 反查 URL: {sec_name} - {title_clean[:30]}...")
            
            # 搜索 cninfo
            payload = {
                "pageNum": 1,
                "pageSize": 10,
                "column": "szse",
                "tabName": "fulltext",
                "plate": "",
                "stock": "",
                "searchkey": title_clean[:50],  # 用标题前50字符搜索
                "secid": "",
                "category": "",
                "trade": "",
                "seDate": "",
                "sortName": "",
                "sortType": "",
                "isHLtitle": "true",
            }
            
            resp = requests.post(CNINFO_API, data=payload, headers=CNINFO_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            announcements = data.get("announcements", [])
            if not announcements:
                logger.warning(f"   ⚠️  未找到公告: {title_clean[:30]}")
                return None
            
            # 匹配公司简称
            for ann in announcements:
                ann_sec_name = ann.get("secName", "")
                ann_title = ann.get("announcementTitle", "")
                adj_url = ann.get("adjunctUrl")
                
                if not adj_url:
                    continue
                
                # 模糊匹配：公司简称相同或标题包含关键词
                if sec_name and ann_sec_name == sec_name:
                    pdf_url = f"http://static.cninfo.com.cn/{adj_url}"
                    logger.info(f"   ✅ 找到 URL: {pdf_url[:60]}...")
                    return pdf_url
            
            # 如果没有精确匹配，取第一个结果
            first_ann = announcements[0]
            adj_url = first_ann.get("adjunctUrl")
            if adj_url:
                pdf_url = f"http://static.cninfo.com.cn/{adj_url}"
                logger.info(f"   ✅ 使用第一个结果: {pdf_url[:60]}...")
                return pdf_url
            
            return None
            
        except Exception as e:
            logger.error(f"   ❌ 反查 URL 失败: {e}")
            return None
    
    def _create_task(self, pdf_url: str) -> Optional[str]:
        """创建 MinerU 解析任务"""
        try:
            task_data = {
                "url": pdf_url,
                "model_version": "vlm",
                "is_ocr": True
            }
            
            response = requests.post(
                f"{self.api_base_url}/task",
                headers=self.headers,
                json=task_data,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"   ❌ 创建任务失败: {response.text}")
                return None
            
            result = response.json()
            if result.get("code") != 0:
                logger.error(f"   ❌ API 错误: {result.get('msg')}")
                return None
            
            task_id = result["data"]["task_id"]
            logger.info(f"   ✅ 任务创建成功: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"   ❌ 创建任务异常: {e}")
            return None
    
    def _poll_task(self, task_id: str, timeout: int = 300, poll_interval: int = 5) -> Optional[str]:
        """轮询任务状态，返回 zip URL"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.api_base_url}/task/{task_id}",
                    headers=self.headers,
                    timeout=10
                )
                
                if response.status_code != 200:
                    logger.error(f"   ❌ 查询状态失败: {response.text}")
                    return None
                
                result = response.json()
                if result.get("code") != 0:
                    logger.error(f"   ❌ API 错误: {result.get('msg')}")
                    return None
                
                data = result["data"]
                state = data.get("state")
                
                if state == "done":
                    zip_url = data.get("full_zip_url")
                    logger.info(f"   ✅ 解析完成")
                    return zip_url
                
                elif state == "failed":
                    error_msg = data.get("err_msg", "未知错误")
                    logger.error(f"   ❌ 解析失败: {error_msg}")
                    return None
                
                elif state == "running":
                    progress = data.get("extract_progress", {})
                    extracted = progress.get("extracted_pages", 0)
                    total = progress.get("total_pages", 0)
                    logger.info(f"   ⏳ 进度: {extracted}/{total} 页...")
                
                else:
                    logger.info(f"   ⏳ 状态: {state}...")
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"   ❌ 轮询异常: {e}")
                time.sleep(poll_interval)
        
        logger.error(f"   ❌ 超时")
        return None
    
    def _download_and_extract(self, zip_url: str, pdf_name: str) -> Optional[str]:
        """下载 zip 并解压，返回 markdown 路径"""
        try:
            logger.info(f"   📥 下载结果...")
            response = requests.get(zip_url, timeout=120)
            
            if response.status_code != 200:
                logger.error(f"   ❌ 下载失败: {response.status_code}")
                return None
            
            # 解压
            output_dir = self.output_dir / pdf_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                zf.extractall(output_dir)
            
            # 查找 markdown 文件
            md_files = list(output_dir.rglob("*.md"))
            if not md_files:
                logger.error(f"   ❌ 未找到 Markdown 文件")
                return None
            
            md_path = md_files[0]
            logger.info(f"   📄 Markdown: {md_path.name}")
            return str(md_path)
            
        except Exception as e:
            logger.error(f"   ❌ 下载解压异常: {e}")
            return None
    
    def parse(self, pdf_path: str) -> Optional[Dict[str, str]]:
        """解析单个 PDF"""
        try:
            pdf_path = Path(pdf_path)
            pdf_name = pdf_path.stem
            
            logger.info(f"📄 解析: {pdf_name[:40]}...")
            
            # 1. 尝试反查 URL（适用于 cninfo 公告）
            pdf_url = self._resolve_cninfo_pdf_url(pdf_path.name)
            
            if pdf_url:
                # 使用 URL 方式
                task_id = self._create_task(pdf_url)
            else:
                # 🔥 使用文件上传方式（适用于研报等本地文件）
                logger.info(f"   📤 使用文件上传方式...")
                task_id = self._create_task_from_file(str(pdf_path))
            
            if not task_id:
                return None
            
            # 3. 轮询状态
            zip_url = self._poll_task(task_id)
            if not zip_url:
                return None
            
            # 4. 下载解压
            md_path = self._download_and_extract(zip_url, pdf_name)
            if not md_path:
                return None
            
            # 读取内容
            content = Path(md_path).read_text(encoding='utf-8')
            images_dir = Path(md_path).parent / "images"
            
            return {
                'markdown': md_path,
                'images': str(images_dir) if images_dir.exists() else None,
                'content': content
            }
            
        except Exception as e:
            logger.error(f"❌ 异常: {pdf_path} - {e}")
            return None
    
    def batch_parse(
        self, 
        pdf_dir: str, 
        skip_existing: bool = True
    ) -> List[Dict[str, str]]:
        """批量解析 PDF"""
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        logger.info(f"📚 找到 {len(pdf_files)} 个 PDF 文件")
        
        parsed_results = []
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            pdf_name = pdf_path.stem
            
            # 检查是否已解析
            if skip_existing:
                existing_md = list((self.output_dir / pdf_name).rglob("*.md"))
                if existing_md:
                    logger.info(f"⏭️  [{idx}/{len(pdf_files)}] 跳过已解析: {pdf_name[:30]}...")
                    parsed_results.append({
                        'markdown': str(existing_md[0]),
                        'images': str(existing_md[0].parent / "images"),
                        'content': existing_md[0].read_text(encoding='utf-8')
                    })
                    continue
            
            # 解析
            logger.info(f"\n📄 [{idx}/{len(pdf_files)}] 解析: {pdf_name[:40]}...")
            result = self.parse(str(pdf_path))
            
            if result:
                parsed_results.append(result)
            
            # 延迟，避免 API 限流
            if idx < len(pdf_files):
                time.sleep(2)
        
        logger.info(f"\n✅ 批量解析完成: {len(parsed_results)}/{len(pdf_files)} 个")
        return parsed_results

    def _create_task_from_file(self, pdf_path: str) -> Optional[str]:
        """通过文件上传创建 MinerU 解析任务"""
        try:
            # 1. 获取上传 URL
            response = requests.post(
                f"{self.api_base_url}/file-urls",
                headers=self.headers,
                json={"file_names": [Path(pdf_path).name]},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"   ❌ 获取上传URL失败: {response.text}")
                return None
            
            result = response.json()
            if result.get("code") != 0:
                logger.error(f"   ❌ API 错误: {result.get('msg')}")
                return None
            
            upload_info = result["data"]["file_urls"][0]
            upload_url = upload_info["url"]
            
            # 2. 上传文件
            with open(pdf_path, 'rb') as f:
                upload_response = requests.put(upload_url, data=f, timeout=120)
            
            if upload_response.status_code not in [200, 201]:
                logger.error(f"   ❌ 文件上传失败: {upload_response.status_code}")
                return None
            
            logger.info(f"   ✅ 文件上传成功")
            
            # 3. 等待系统自动创建任务并返回 task_id
            # MinerU 会自动扫描上传的文件并创建任务
            task_id = upload_info.get("task_id")
            if task_id:
                return task_id
            
            # 如果没有立即返回 task_id，需要轮询获取
            time.sleep(3)
            return self._get_task_id_by_filename(Path(pdf_path).name)
            
        except Exception as e:
            logger.error(f"   ❌ 文件上传异常: {e}")
            return None