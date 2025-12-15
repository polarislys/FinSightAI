"""
测试 AkShare 工具模块
"""
import sys
sys.path.append('/home/nl/disk_8T/lys/FinSightAI')

from src.tools.akshare_tool import akshare_tool

def test_stock_realtime():
    """测试实时行情"""
    print("=" * 60)
    print("测试1: 获取贵州茅台实时行情")
    print("=" * 60)
    result = akshare_tool.get_stock_realtime("600519")
    print(akshare_tool.format_response(result))

def test_financial_metrics():
    """测试财务指标"""
    print("=" * 60)
    print("测试2: 获取贵州茅台财务指标")
    print("=" * 60)
    result = akshare_tool.get_financial_metrics("600519")
    print(akshare_tool.format_response(result))

def test_stock_news():
    """测试新闻获取"""
    print("=" * 60)
    print("测试3: 获取贵州茅台新闻")
    print("=" * 60)
    result = akshare_tool.get_stock_news("600519", limit=5)
    print(akshare_tool.format_response(result))

def test_plot_kline():
    """测试K线图绘制"""
    print("=" * 60)
    print("测试4: 绘制贵州茅台K线图")
    print("=" * 60)
    result = akshare_tool.plot_kline("600519", days=30)
    if "error" not in result:
        print(f"✅ K线图已保存到: {result['chart_path']}")
    else:
        print(akshare_tool.format_response(result))

if __name__ == "__main__":
    test_stock_realtime()
    test_financial_metrics()
    # test_stock_news()
    # test_plot_kline()