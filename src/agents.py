"""
Multi-agent supervisor pattern for SellerSense.

Implements a tool-calling loop where the LLM acts as a supervisor that can
decide to:
1. Call tools to gather information (assess_item, get_context, etc.)
2. Synthesize a response from the tool outputs
3. Hand off to a specialist agent for specific tasks

The supervisor orchestrates tool calls in a ReAct-style loop, bounded by
max_iterations to prevent runaway loops. Each iteration either calls tools
or produces a final answer.

LangSmith: all agents are traceable via @traceable decorators.
"""

import sys
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool
except ImportError:
    pass

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type="chain"):
        def decorator(func):
            return func
        return decorator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logger import get_logger
from tools import ALL_TOOLS

logger = get_logger(__name__, extra_data={"module": "agents"})


# ---- Agent State ----

class AgentState(TypedDict):
    messages: list
    item_id: Optional[str]
    as_of_date: Optional[str]
    final_answer: Optional[str]
    iteration: int
    max_iterations: int


# ---- Specialist Response Schemas ----

class InventoryAnalysis(BaseModel):
    """Structured analysis of an inventory item."""
    item_id: str
    risk_level: str = Field(description="stockout_risk, overstock, or healthy")
    current_stock: int
    days_of_cover: float | None
    suggested_order_qty: int
    context_summary: str = Field(description="festival/promo/seasonal context")
    trend_summary: str = Field(description="recent demand trend")
    recommendation: str = Field(description="plain-language recommendation")
    confidence: float = Field(description="0-1 confidence in the recommendation")


class SupervisorDecision(BaseModel):
    """What the supervisor decides to do next."""
    action: Literal["call_tool", "respond", "handoff"]
    tool_name: Optional[str] = Field(description="tool to call if action is call_tool")
    tool_args: Optional[dict] = Field(description="arguments for the tool")
    response: Optional[str] = Field(description="final response if action is respond")
    handoff_to: Optional[str] = Field(description="agent to hand off to if action is handoff")


# ---- Supervisor Agent ----

@traceable(name="supervisor_agent", run_type="chain")
def supervisor_agent(
    llm,
    messages: list,
    tools: list = None,
    max_iterations: int = 5,
    as_of_date: str = "",
) -> str:
    """
    Run a tool-calling supervisor loop.
    
    The LLM decides whether to call tools or provide a final answer.
    Tools are called and results are fed back until the LLM produces
    a final response or the iteration limit is reached.
    
    Args:
        llm: LangChain chat model with tool-calling support
        messages: List of conversation messages
        tools: List of tools available to the agent (default: ALL_TOOLS)
        max_iterations: Maximum tool-calling iterations
        as_of_date: Current date for context
    
    Returns:
        Final answer string
    """
    if tools is None:
        tools = ALL_TOOLS

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    iteration = 0
    conversation = list(messages)
    
    logger.info("Starting supervisor loop", extra={"extra_data": {
        "n_tools": len(tools),
        "max_iterations": max_iterations,
    }})

    while iteration < max_iterations:
        iteration += 1
        
        # Get LLM response
        response = llm_with_tools.invoke(conversation)
        conversation.append(response)
        
        # If no tool calls, we have our final answer
        if not response.tool_calls:
            logger.info("Supervisor produced final answer", extra={"extra_data": {
                "iteration": iteration,
            }})
            return response.content
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info("Supervisor calling tool", extra={"extra_data": {
                "tool": tool_name,
                "args": tool_args,
            }})
            
            # Find and execute the tool
            tool_func = next((t for t in tools if t.name == tool_name), None)
            if tool_func is None:
                result = f"Error: tool '{tool_name}' not found"
            else:
                try:
                    result = tool_func.invoke(tool_args)
                except Exception as e:
                    result = f"Error calling {tool_name}: {str(e)}"
            
            # Add tool result to conversation
            conversation.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            ))
    
    # If we hit the iteration limit, ask for a final answer
    logger.warning("Supervisor hit iteration limit", extra={"extra_data": {
        "iteration": iteration,
    }})
    conversation.append(HumanMessage(content="Please provide your final answer now based on the information gathered."))
    response = llm_with_tools.invoke(conversation)
    return response.content


# ---- Specialist Agents ----

@traceable(name="inventory_analyst_agent", run_type="chain")
def inventory_analyst_agent(llm, item_id: str, as_of_date: str) -> dict:
    """
    Specialist agent that produces a structured inventory analysis.
    
    Uses tools to gather data, then produces a structured analysis.
    """
    tools = [t for t in ALL_TOOLS if t.name in [
        "assess_inventory_item", "get_item_context", 
        "get_on_order_status", "get_demand_trend",
    ]]
    
    messages = [
        HumanMessage(content=(
            f"Analyze inventory item {item_id} as of {as_of_date}. "
            "Call the available tools to gather current stock status, "
            "context signals (festivals, promos), on-order status, and "
            "demand trends. Then produce a structured analysis."
        ))
    ]
    
    # Use the tools directly for structured data gathering
    assessment = None
    context = None
    on_order = None
    trend = None
    
    for t in tools:
        try:
            if t.name == "assess_inventory_item":
                assessment = t.invoke({"item_id": item_id, "as_of_date": as_of_date})
            elif t.name == "get_item_context":
                context = t.invoke({"item_id": item_id, "as_of_date": as_of_date})
            elif t.name == "get_on_order_status":
                on_order = t.invoke({"item_id": item_id, "as_of_date": as_of_date})
            elif t.name == "get_demand_trend":
                trend = t.invoke({"item_id": item_id})
        except Exception as e:
            logger.warning(f"Tool {t.name} failed: {e}")

    # Synthesize analysis with LLM
    context_str = str(context) if context else "no context available"
    trend_str = str(trend) if trend else "no trend data"
    on_order_str = str(on_order) if on_order else "no order data"
    
    prompt = (
        f"Based on this inventory data for item {item_id}:\n"
        f"Assessment: {assessment}\n"
        f"Context: {context_str}\n"
        f"On Order: {on_order_str}\n"
        f"Trend: {trend_str}\n\n"
        "Produce a structured inventory analysis with a clear recommendation."
    )
    
    try:
        structured_llm = llm.with_structured_output(InventoryAnalysis)
        return structured_llm.invoke(prompt)
    except Exception:
        # Fallback: return raw data
        return {
            "item_id": item_id,
            "assessment": assessment,
            "context": context,
            "on_order": on_order,
            "trend": trend,
        }


@traceable(name="chat_agent", run_type="chain")
def chat_agent(llm, message: str, consumption: dict, context: dict, as_of_date: str) -> str:
    """
    Chat agent that answers natural language questions about inventory.
    
    Uses the supervisor loop to decide whether tools are needed.
    """
    system_prompt = (
        "You are an inventory assistant for a small retail store. "
        "Answer the shop owner's question concisely. "
        "Use the available tools to look up specific data when needed. "
        "Always ground your answers in the data, never invent numbers."
    )
    
    messages = [
        HumanMessage(content=f"Date: {as_of_date}\n\n{message}")
    ]
    
    return supervisor_agent(
        llm, messages, tools=ALL_TOOLS,
        max_iterations=3, as_of_date=as_of_date,
    )


# Export for use in dashboard and graph
__all__ = [
    "supervisor_agent",
    "inventory_analyst_agent", 
    "chat_agent",
    "ALL_TOOLS",
]
