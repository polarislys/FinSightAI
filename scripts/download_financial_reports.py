"""
下载券商研报（东方财富网）
适配 FinSightAI 项目结构
"""
import asyncio
import httpx
import os
import random
import json
import time
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import List, Dict

class ResearchReportCrawler:
    """券商研报爬虫（东方财富）"""
    
    def __init__(self, save_dir="./data/raw_pdfs/financial_reports"):
        self.base_url = "https://reportapi.eastmoney.com/report/list"
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 伪造 User-Agent
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://data.eastmoney.com/"
        }
        
        self.downloaded_count = 0
        self.failed_count = 0

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def fetch_report_list(
        self, 
        page: int = 1, 
        page_size: int = 10,
        q_type: str = "0",  # 0=个股研报, 1=行业研报
        begin_time: str = "2024-01-01",
        end_time: str = "2025-12-31"
    ) -> List[Dict]:
        """
        获取研报列表元数据
        
        Args:
            page: 页码
            page_size: 每页数量
            q_type: 0=个股研报, 1=行业研报
            begin_time: 开始时间
            end_time: 结束时间
        """
        params = {
            "cb": f"datatable{random.randint(1000000, 9999999)}",
            "industryCode": "*",
            "pageSize": page_size,
            "industry": "*",
            "rating": "*",
            "ratingChange": "*",
            "beginTime": begin_time,
            "endTime": end_time,
            "pageNo": page,
            "fields": "",
            "qType": q_type,
            "orgCode": "",
            "code": "",
            "rcode": "",
            "p": page,
            "pageNum": page,
            "_": str(int(time.time() * 1000))
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    self.base_url, 
                    params=params, 
                    headers=self.headers
                )
                
                # 解析 JSONP 格式
                text = response.text
                start_idx = text.find('(') + 1
                end_idx = text.rfind(')')
                json_str = text[start_idx:end_idx]
                data = json.loads(json_str)
                
                return data.get("data", [])
        except Exception as e:
            print(f"❌ 获取列表失败 (页{page}): {e}")
            return []

    def clean_filename(self, title: str, max_length: int = 100) -> str:
        """清洗文件名，移除非法字符"""
        # 移除特殊字符
        safe_chars = "".join(
            c for c in title 
            if c.isalnum() or c in (' ', '-', '_', '（', '）', '、', '：')
        ).strip()
        
        # 限制长度
        if len(safe_chars) > max_length:
            safe_chars = safe_chars[:max_length]
        
        return safe_chars if safe_chars else "untitled"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    async def download_pdf(self, report: Dict) -> bool:
        """
        下载单个 PDF 文件
        
        Args:
            report: 研报元数据字典
        """
        title = report.get("title", "Untitled")
        info_code = report.get("infoCode")
        
        if not info_code:
            print(f"  ⏭️  跳过（无infoCode）: {title[:50]}")
            return False
        
        # 构造 PDF URL（东方财富研报链接规则）
        # 注意：实际URL可能需要根据网站最新结构调整
        pdf_url = f"https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf"
        
        # 清洗文件名
        safe_title = self.clean_filename(title)
        file_path = os.path.join(self.save_dir, f"{safe_title}.pdf")

        # 检查是否已存在
        if os.path.exists(file_path):
            print(f"  ✅ 已存在: {safe_title[:60]}")
            return True

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(pdf_url, headers=self.headers)
                
                if response.status_code == 200:
                    # 检查是否真的是PDF（有些链接可能失效）
                    content_type = response.headers.get("content-type", "")
                    if "pdf" not in content_type.lower() and len(response.content) < 1000:
                        print(f"  ⚠️  非PDF或文件过小: {title[:50]}")
                        self.failed_count += 1
                        return False
                    
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    
                    self.downloaded_count += 1
                    print(f"  ✅ 下载成功 ({len(response.content)//1024}KB): {safe_title[:60]}")
                    return True
                else:
                    print(f"  ❌ 下载失败 (HTTP {response.status_code}): {title[:50]}")
                    self.failed_count += 1
                    return False
                    
        except Exception as e:
            print(f"  ❌ 下载异常: {title[:50]} - {e}")
            self.failed_count += 1
            return False

    async def run(
        self, 
        pages: int = 3, 
        page_size: int = 10,
        q_type: str = "1",  # 默认下载行业研报
        begin_time: str = "2024-01-01"
    ):
        """
        主执行逻辑
        
        Args:
            pages: 爬取页数
            page_size: 每页数量
            q_type: 0=个股研报, 1=行业研报
            begin_time: 开始时间
        """
        print("=" * 70)
        print(f"🚀 开始下载{'行业' if q_type == '1' else '个股'}研报")
        print(f"📁 保存路径: {self.save_dir}")
        print(f"📄 计划下载: {pages} 页 × {page_size} 条/页 = {pages * page_size} 份研报")
        print("=" * 70)
        
        all_tasks = []
        
        for page in range(1, pages + 1):
            print(f"\n📖 正在获取第 {page}/{pages} 页元数据...")
            
            reports = await self.fetch_report_list(
                page=page,
                page_size=page_size,
                q_type=q_type,
                begin_time=begin_time
            )
            
            if not reports:
                print(f"  ⚠️  第 {page} 页无数据，跳过")
                continue
            
            print(f"  ✅ 获取到 {len(reports)} 条研报元数据")
            
            # 为每个研报创建下载任务
            for i, report in enumerate(reports, 1):
                title = report.get("title", "Unknown")
                print(f"  [{i}/{len(reports)}] {title[:70]}")
                
                # 添加随机延迟，避免被封IP
                await asyncio.sleep(random.uniform(0.5, 1.5))
                all_tasks.append(self.download_pdf(report))
            
            # 页面间延迟
            if page < pages:
                await asyncio.sleep(random.uniform(2, 4))
        
        # 并发下载所有PDF
        print(f"\n⏳ 开始批量下载 {len(all_tasks)} 个文件...")
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # 统计结果
        print("\n" + "=" * 70)
        print("📊 下载统计")
        print("=" * 70)
        print(f"✅ 成功下载: {self.downloaded_count} 份")
        print(f"❌ 下载失败: {self.failed_count} 份")
        print(f"📁 保存路径: {self.save_dir}")
        print("=" * 70)


async def main():
    """主函数"""
    crawler = ResearchReportCrawler(
        save_dir="./data/raw_pdfs/research_reports"
    )
    
    # 下载行业研报（推荐用于学习研报写作风格）
    await crawler.run(
        pages=3,           # 下载3页
        page_size=10,      # 每页10条
        q_type="1",        # 1=行业研报（更适合学习分析框架）
        begin_time="2024-01-01"
    )
    
    # 如果还想下载个股研报，可以再运行一次
    # await crawler.run(pages=2, page_size=10, q_type="0", begin_time="2024-01-01")


if __name__ == "__main__":
    # 安装依赖提示
    try:
        import httpx
        import tenacity
    except ImportError:
        print("❌ 缺少依赖，请先安装：")
        print("pip install httpx tenacity")
        exit(1)
    
    # 运行爬虫
    asyncio.run(main())