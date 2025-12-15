"""
意图识别器
使用 Qwen2.5-0.5B 小模型进行快速意图分类
"""
from openai import OpenAI
import os
import json
from typing import Dict

class IntentDetector:
    """意图识别器（使用0.5B小模型）"""
    
    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1"
        )
        # 使用 0.5B 小模型（快速、便宜）
        self.model = "Qwen/Qwen2.5-0.5B-Instruct"
        
        # 意图定义
        self.intents = {
            "stock_price": "查询股票价格、行情、涨跌",
            "financial_metrics": "查询财务指标（PE、PB、ROE、毛利率等）",
            "company_events": "查询公司事件（投资、中标、合作等）",
            "risk_check": "查询风险（诉讼、担保、质押等）",
            "industry_analysis": "查询行业分析、趋势、竞争格局",
            "news_sentiment": "查询新闻、舆情",
            "general_qa": "一般性金融问答"
        }
    
    def detect(self, query: str) -> Dict:
        """
        检测用户意图
        
        Args:
            query: 用户查询
            
        Returns:
            {
                "intent": str,           # 意图类型
                "confidence": float,     # 置信度
                "entities": Dict         # 提取的实体（股票代码、公司名等）
            }
        """
        # 构建提示词（Few-Shot）
        system_prompt = """你是一个金融查询意图分类器。
请分析用户问题，返回JSON格式的分类结果。

可选意图类型：
- stock_price: 查询股价、行情
- financial_metrics: 查询财务指标
- company_events: 查询公司事件
- risk_check: 查询风险
- industry_analysis: 查询行业分析
- news_sentiment: 查询新闻舆情
- general_qa: 一般问答

返回格式：
{"intent": "意图类型", "stock_code": "股票代码或null", "company_name": "公司名或null"}

示例：
问题："贵州茅台现在多少钱？"
输出：{"intent": "stock_price", "stock_code": "600519", "company_name": "贵州茅台"}

问题："新能源汽车行业趋势如何？"
输出：{"intent": "industry_analysis", "stock_code": null, "company_name": null}"""

        user_prompt = f"问题：{query}\n输出："
        
        # 调用小模型
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # 解析 JSON
        try:
            result = json.loads(result_text)
            intent = result.get("intent", "general_qa")
            
            return {
                "intent": intent,
                "confidence": 0.9,  # 小模型置信度可以固定
                "entities": {
                    "stock_code": result.get("stock_code"),
                    "company_name": result.get("company_name")
                }
            }
        except json.JSONDecodeError:
            # 如果解析失败，使用关键词匹配作为后备
            return self._fallback_detection(query)
    
    def _fallback_detection(self, query: str) -> Dict:
        """关键词匹配作为后备方案"""
        # 股价关键词
        if any(kw in query for kw in ["股价", "价格", "多少钱", "涨跌", "行情"]):
            return {"intent": "stock_price", "confidence": 0.7, "entities": {}}
        
        # 财务指标
        if any(kw in query for kw in ["市盈率", "PE", "PB", "ROE", "毛利率", "估值"]):
            return {"intent": "financial_metrics", "confidence": 0.7, "entities": {}}
        
        # 风险
        if any(kw in query for kw in ["诉讼", "担保", "质押", "风险", "违规"]):
            return {"intent": "risk_check", "confidence": 0.7, "entities": {}}
        
        # 行业分析
        if any(kw in query for kw in ["行业", "趋势", "竞争", "格局", "增速"]):
            return {"intent": "industry_analysis", "confidence": 0.7, "entities": {}}
        
        # 默认
        return {"intent": "general_qa", "confidence": 0.5, "entities": {}}