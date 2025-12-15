"""
对话管理器
使用 LangChain Checkpointer 管理对话历史
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
import os

class ConversationState(TypedDict):
    """对话状态"""
    messages: Annotated[List[BaseMessage], "对话消息列表"]
    current_query: str
    rewritten_query: str

class ConversationManager:
    """对话管理器（基于 LangChain Checkpointer）"""
    
    def __init__(self):
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model="Qwen/Qwen2.5-7B-Instruct",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            temperature=0.1
        )
        
        # 创建内存检查点（生产环境可换成 SQLite/Redis）
        self.checkpointer = MemorySaver()
        
        # 构建对话图
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建对话流程图"""
        workflow = StateGraph(ConversationState)
        
        # 添加节点
        workflow.add_node("rewrite_query", self._rewrite_query_node)
        
        # 添加边
        workflow.add_edge(START, "rewrite_query")
        workflow.add_edge("rewrite_query", END)
        
        # 编译图（带检查点）
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _rewrite_query_node(self, state: ConversationState) -> ConversationState:
        """查询重写节点"""
        messages = state["messages"]
        current_query = state["current_query"]
        
        # 如果没有历史，直接返回
        if len(messages) <= 1:
            state["rewritten_query"] = current_query
            return state
        
        # 构建重写提示
        system_prompt = """你是查询重写助手。
根据历史对话，将当前问题改写为完整、独立的问题。
只输出改写后的问题，不要解释。

规则：
1. 解决代词指代（"它"→具体公司名）
2. 补充缺失上下文
3. 保持原意图"""
        
        # 构建历史上下文
        history_text = "\n".join([
            f"{'用户' if isinstance(msg, HumanMessage) else 'AI'}：{msg.content}"
            for msg in messages[-4:]  # 最近4轮
        ])
        
        user_prompt = f"""历史对话：
{history_text}

当前问题：{current_query}

改写后的问题："""
        
        # 调用 LLM
        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        rewritten = response.content.strip()
        
        print(f"🔄 查询重写: '{current_query}' → '{rewritten}'")
        
        state["rewritten_query"] = rewritten
        return state
    
    def process_query(
        self, 
        query: str, 
        session_id: str = "default"
    ) -> Dict:
        """
        处理查询（带历史回溯）
        
        Args:
            query: 用户查询
            session_id: 会话ID（用于区分不同用户）
            
        Returns:
            {
                "original_query": str,
                "rewritten_query": str,
                "messages": List[BaseMessage]
            }
        """
        # 配置（指定会话ID）
        config = {"configurable": {"thread_id": session_id}}
        
        # 获取当前状态
        current_state = self.graph.get_state(config)
        
        # 构建新状态
        messages = current_state.values.get("messages", []) if current_state.values else []
        messages.append(HumanMessage(content=query))
        
        initial_state = {
            "messages": messages,
            "current_query": query,
            "rewritten_query": ""
        }
        
        # 执行图（自动保存检查点）
        result = self.graph.invoke(initial_state, config)
        
        return {
            "original_query": query,
            "rewritten_query": result["rewritten_query"],
            "messages": result["messages"]
        }
    
    def add_ai_response(self, response: str, session_id: str = "default"):
        """
        添加 AI 回复到历史
        
        Args:
            response: AI 回复内容
            session_id: 会话ID
        """
        config = {"configurable": {"thread_id": session_id}}
        current_state = self.graph.get_state(config)
        
        if current_state.values:
            messages = current_state.values.get("messages", [])
            messages.append(AIMessage(content=response))
            
            # 更新状态
            self.graph.update_state(
                config,
                {"messages": messages}
            )
    
    def clear_history(self, session_id: str = "default"):
        """清空会话历史"""
        config = {"configurable": {"thread_id": session_id}}
        # LangGraph 会自动管理，这里可以重置状态
        self.graph.update_state(config, {"messages": []})