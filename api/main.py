"""
FastAPI 入口 - 暴露 Sprint 1/2 检索与问答接口
"""
import sys
sys.path.insert(0, '.')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 请求/响应模型
# ============================================================

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    top_k: int = 5


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="FinSightAI API",
    description="金融智能问答系统 API",
    version="0.1.0"
)

# 全局服务实例（启动时初始化）
rag_service = None


@app.on_event("startup")
async def startup():
    """服务启动时初始化 RAGService（只做一次）"""
    global rag_service
    logger.info("🚀 启动 FinSightAI API...")
    
    from src.services.rag_service import RAGService
    rag_service = RAGService()
    
    logger.info("✅ API 启动完成")


@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "message": "FinSightAI API is running"}


@app.post("/v1/retrieve")
async def retrieve(request: RetrieveRequest) -> Dict[str, Any]:
    """
    检索接口（调试用）
    
    返回四路检索结果对比：vector, bm25, rrf, rrf_rerank
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    try:
        result = rag_service.retrieve(request.query, request.top_k)
        return result
    except Exception as e:
        logger.error(f"❌ 检索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    问答接口
    
    输入 OpenAI 格式的 messages，返回 answer 和 source_documents
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        result = rag_service.chat(messages, request.top_k)
        return result
    except Exception as e:
        logger.error(f"❌ 问答失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)