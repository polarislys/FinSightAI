"""
爬取巨潮资讯网上市公司年报/研报
网站：http://www.cninfo.com.cn
"""
import os
import time
import requests
from datetime import datetime

# 配置
SAVE_DIR = "./data/raw_pdfs"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "http://www.cninfo.com.cn/new/disclosure",
}

# 巨潮资讯网 API
CNINFO_API = "http://www.cninfo.com.cn/new/hisAnnouncement/query"


def search_reports(stock_code: str, report_type: str = "annual", page: int = 1, page_size: int = 30):
    """
    搜索公告/年报
    report_type: annual(年报), semi(半年报), quarterly(季报), research(研报)
    """
    # 公告类型映射
    category_map = {
        "annual": "category_ndbg_szsh",      # 年度报告
        "semi": "category_bndbg_szsh",       # 半年度报告  
        "quarterly": "category_sjdbg_szsh",  # 季度报告
        "all": ""                            # 所有公告
    }
    
    payload = {
        "pageNum": page,
        "pageSize": page_size,
        "column": "szse",  # 深交所，可改为 "sse" 上交所
        "tabName": "fulltext",
        "plate": "",
        "stock": stock_code,
        "searchkey": "",
        "secid": "",
        "category": category_map.get(report_type, ""),
        "trade": "",
        "seDate": "",  # 可设置日期范围，如 "2023-01-01~2024-01-01"
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    
    try:
        resp = requests.post(CNINFO_API, data=payload, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("announcements", [])
    except Exception as e:
        print(f"搜索失败: {e}")
        return []


def download_pdf(announcement: dict, save_subdir: str = "annual_reports"):
    """下载单个 PDF 文件"""
    adj_url = announcement.get("adjunctUrl")
    title = announcement.get("announcementTitle", "unknown")
    sec_name = announcement.get("secName", "unknown")  # 公司简称
    
    if not adj_url:
        print(f"  跳过（无附件）: {title}")
        return False
    
    # 构建下载 URL
    pdf_url = f"http://static.cninfo.com.cn/{adj_url}"
    
    # 清理文件名
    safe_title = "".join(c for c in f"{sec_name}_{title}" if c.isalnum() or c in "（）()_- ")[:100]
    save_path = os.path.join(SAVE_DIR, save_subdir, f"{safe_title}.pdf")
    
    # 检查是否已存在
    if os.path.exists(save_path):
        print(f"  已存在: {safe_title}")
        return True
    
    try:
        print(f"  下载中: {safe_title}")
        resp = requests.get(pdf_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        
        print(f"  完成: {save_path}")
        time.sleep(1)  # 礼貌性延迟
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        return False


def batch_download(stock_codes: list, report_type: str = "annual", max_per_stock: int = 5):
    """
    批量下载多只股票的报告
    stock_codes: 股票代码列表，如 ["000001", "600519"]
    """
    subdir_map = {
        "annual": "annual_reports",
        "semi": "semi_annual_reports",
        "quarterly": "quarterly_reports",
    }
    save_subdir = subdir_map.get(report_type, "other_reports")
    
    for code in stock_codes:
        print(f"\n{'='*50}")
        print(f"正在处理: {code}")
        print(f"{'='*50}")
        
        reports = search_reports(code, report_type)
        if not reports:
            print(f"  未找到报告")
            continue
        
        for i, report in enumerate(reports[:max_per_stock]):
            download_pdf(report, save_subdir)
        
        time.sleep(2)  # 股票间延迟


def download_by_keyword(keyword: str, save_subdir: str = "research_reports", max_count: int = 20):
    """按关键词搜索并下载（适合研报）"""
    payload = {
        "pageNum": 1,
        "pageSize": max_count,
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": "",
        "searchkey": keyword,
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": "",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    
    try:
        resp = requests.post(CNINFO_API, data=payload, headers=HEADERS, timeout=30)
        data = resp.json()
        announcements = data.get("announcements", [])
        
        print(f"找到 {len(announcements)} 条结果")
        for ann in announcements:
            download_pdf(ann, save_subdir)
            
    except Exception as e:
        print(f"搜索失败: {e}")


if __name__ == "__main__":
    # 【新能源产业链】爬取30-50份PDF用于技术验证
    # 由于年报API可能限制，改为多关键词搜索策略
    
    print("=" * 60)
    print("开始下载新能源产业链相关PDF文档")
    print("=" * 60)
    
    # 扩展关键词列表以覆盖新能源全产业链
    keywords = [
        "新能源汽车",
        "动力电池", 
        "光伏产业",
        "锂电池",
        "储能",
        "充电桩",
        "电池材料",
        "逆变器",
        "风电",
        "氢能源",
    ]
    
    for keyword in keywords:
        print(f"\n{'='*50}")
        print(f"正在搜索关键词: {keyword}")
        print(f"{'='*50}")
        download_by_keyword(keyword, save_subdir="research_reports", max_count=5)
        time.sleep(2)  # 关键词间延迟
    
    print("\n" + "=" * 60)
    print("✅ 下载完成！")
    print("存储路径: ./data/raw_pdfs/")
    print("=" * 60)