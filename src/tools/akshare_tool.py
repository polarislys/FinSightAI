"""
AkShare 工具模块
提供股票行情、财务数据、新闻等查询功能
"""
import akshare as ak
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端

class AkShareTool:
    """AkShare 数据获取工具"""
    
    def __init__(self):
        """初始化 AkShare 工具"""
        self.name = "akshare_tool"
        
    def get_stock_hist(
        self, 
        symbol: str, 
        period: str = "daily",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        获取股票历史行情数据
        
        Args:
            symbol: 股票代码（如 "600519" 表示贵州茅台）
            period: 周期，可选 "daily", "weekly", "monthly"
            start_date: 开始日期，格式 "20240101"
            end_date: 结束日期，格式 "20241231"
            
        Returns:
            包含历史行情数据的字典
        """
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date or "20200101",
                end_date=end_date or datetime.now().strftime("%Y%m%d"),
                adjust="qfq"  # 前复权
            )
            
            if df.empty:
                return {"error": f"未找到股票 {symbol} 的历史数据"}
            
            # 获取最新数据
            latest = df.iloc[-1]
            
            return {
                "symbol": symbol,
                "latest_date": str(latest['日期']),
                "latest_close": float(latest['收盘']),
                "latest_open": float(latest['开盘']),
                "latest_high": float(latest['最高']),
                "latest_low": float(latest['最低']),
                "latest_volume": int(latest['成交量']),
                "change_pct": float(latest['涨跌幅']),
                "total_records": len(df),
                "data": df.to_dict('records')[-10:]  # 返回最近10条
            }
        except Exception as e:
            return {"error": f"获取历史行情失败: {str(e)}"}
    
    def get_stock_realtime(self, symbol: str) -> Dict:
        """
        获取股票实时行情
        
        Args:
            symbol: 股票代码
            
        Returns:
            实时行情数据字典
        """
        try:
            # 获取最近一天的数据作为实时数据
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=(datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq"
            )
            
            if df.empty:
                return {"error": f"未找到股票 {symbol} 的实时数据"}
            
            latest = df.iloc[-1]
            
            return {
                "symbol": symbol,
                "date": str(latest['日期']),
                "current_price": float(latest['收盘']),
                "open": float(latest['开盘']),
                "high": float(latest['最高']),
                "low": float(latest['最低']),
                "volume": int(latest['成交量']),
                "change_pct": float(latest['涨跌幅']),
                "change_amount": float(latest['涨跌额']),
                "turnover_rate": float(latest['换手率'])
            }
        except Exception as e:
            return {"error": f"获取实时行情失败: {str(e)}"}
    
    def get_financial_metrics(self, symbol: str) -> Dict:
        """
        获取股票财务指标（PE、PB、ROE等）
        
        Args:
            symbol: 股票代码
            
        Returns:
            财务指标字典
        """
        try:
            # 获取个股信息
            df = ak.stock_individual_info_em(symbol=symbol)
            
            if df.empty:
                return {"error": f"未找到股票 {symbol} 的财务指标"}
            
            # 转换为字典
            info_dict = dict(zip(df['item'], df['value']))
            
            return {
                "symbol": symbol,
                "stock_name": info_dict.get('股票简称', 'N/A'),
                "total_market_cap": info_dict.get('总市值', 'N/A'),
                "pe_ratio": info_dict.get('市盈率-动态', 'N/A'),
                "pb_ratio": info_dict.get('市净率', 'N/A'),
                "roe": info_dict.get('净资产收益率', 'N/A'),
                "gross_margin": info_dict.get('毛利率', 'N/A'),
                "debt_ratio": info_dict.get('资产负债率', 'N/A'),
                "raw_data": info_dict
            }
        except Exception as e:
            return {"error": f"获取财务指标失败: {str(e)}"}
    
    def get_financial_report(
        self, 
        symbol: str, 
        report_type: str = "利润表"
    ) -> Dict:
        """
        获取财务报表数据
        
        Args:
            symbol: 股票代码
            report_type: 报表类型，可选 "利润表", "资产负债表", "现金流量表"
            
        Returns:
            财务报表数据字典
        """
        try:
            if report_type == "利润表":
                df = ak.stock_financial_report_sina(stock=symbol, symbol="利润表")
            elif report_type == "资产负债表":
                df = ak.stock_financial_report_sina(stock=symbol, symbol="资产负债表")
            elif report_type == "现金流量表":
                df = ak.stock_financial_report_sina(stock=symbol, symbol="现金流量表")
            else:
                return {"error": f"不支持的报表类型: {report_type}"}
            
            if df.empty:
                return {"error": f"未找到股票 {symbol} 的{report_type}数据"}
            
            return {
                "symbol": symbol,
                "report_type": report_type,
                "data": df.to_dict('records')
            }
        except Exception as e:
            return {"error": f"获取财务报表失败: {str(e)}"}
    
    def get_stock_news(self, symbol: str, limit: int = 10) -> Dict:
        """
        获取个股新闻
        
        Args:
            symbol: 股票代码
            limit: 返回新闻数量
            
        Returns:
            新闻列表字典
        """
        try:
            df = ak.stock_news_em(symbol=symbol)
            
            if df.empty:
                return {"error": f"未找到股票 {symbol} 的新闻"}
            
            # 只返回最新的 limit 条
            news_list = df.head(limit).to_dict('records')
            
            return {
                "symbol": symbol,
                "total_count": len(df),
                "news": news_list
            }
        except Exception as e:
            return {"error": f"获取新闻失败: {str(e)}"}
    
    def plot_kline(
        self, 
        symbol: str, 
        days: int = 90,
        save_path: Optional[str] = None
    ) -> Dict:
        """绘制K线图"""
        try:
            import os
            
            # 获取历史数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq"
            )
            
            if df.empty:
                return {"error": f"未找到股票 {symbol} 的数据"}
            
            # 配置中文字体（避免警告）
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 绘制K线图
            plt.figure(figsize=(12, 6))
            plt.plot(df['日期'], df['收盘'], label='Close Price', linewidth=2)
            plt.fill_between(df['日期'], df['最低'], df['最高'], alpha=0.3, label='Price Range')
            
            plt.title(f'Stock {symbol} - Last {days} Days', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price (CNY)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图片（自动创建目录）
            if save_path is None:
                save_dir = "./data/charts"
                os.makedirs(save_dir, exist_ok=True)  # 创建目录
                save_path = f"{save_dir}/kline_{symbol}_{datetime.now().strftime('%Y%m%d')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "symbol": symbol,
                "chart_path": save_path,
                "days": days,
                "data_points": len(df)
            }
        except Exception as e:
            return {"error": f"绘制K线图失败: {str(e)}"}
    
    def format_response(self, data: Dict) -> str:
        """
        格式化返回结果为可读文本
        
        Args:
            data: 数据字典
            
        Returns:
            格式化的文本
        """
        if "error" in data:
            return f"❌ {data['error']}"
        
        # 根据不同类型格式化输出
        if "current_price" in data:
            return f"""
📊 股票 {data['symbol']} 实时行情（{data['date']}）
当前价格: ¥{data['current_price']:.2f}
涨跌幅: {data['change_pct']:.2f}%
涨跌额: ¥{data['change_amount']:.2f}
今日开盘: ¥{data['open']:.2f}
最高价: ¥{data['high']:.2f}
最低价: ¥{data['low']:.2f}
成交量: {data['volume']:,}
换手率: {data['turnover_rate']:.2f}%
"""
        elif "pe_ratio" in data:
            return f"""
📈 股票 {data['symbol']} ({data['stock_name']}) 财务指标
总市值: {data['total_market_cap']}
市盈率(PE): {data['pe_ratio']}
市净率(PB): {data['pb_ratio']}
净资产收益率(ROE): {data['roe']}
毛利率: {data['gross_margin']}
资产负债率: {data['debt_ratio']}
"""
        elif "news" in data:
            news_text = f"📰 股票 {data['symbol']} 最新新闻（共{data['total_count']}条）\n\n"
            for i, news in enumerate(data['news'][:5], 1):
                news_text += f"{i}. {news.get('新闻标题', 'N/A')}\n"
                news_text += f"   时间: {news.get('发布时间', 'N/A')}\n\n"
            return news_text
        
        return str(data)


# 创建全局实例
akshare_tool = AkShareTool()