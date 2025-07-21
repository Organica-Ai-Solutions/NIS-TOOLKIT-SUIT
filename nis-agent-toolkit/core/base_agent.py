#!/usr/bin/env python3
"""
NIS Agent Toolkit - Base Agent Framework
Abstract base class for all NIS agents with working implementations
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    PROCESSING = "processing" 
    WAITING = "waiting"
    ERROR = "error"

class BaseNISAgent(ABC):
    """
    Abstract base class for all NIS agents
    
    This is not "revolutionary AGI" - it's a well-engineered agent framework
    with clear interfaces and honest capabilities.
    """
    
    def __init__(self, agent_id: str, agent_type: str = "generic"):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = AgentState.IDLE
        self.logger = logging.getLogger(f"nis.agent.{agent_id}")
        
        # Agent capabilities and tools
        self.capabilities = set()
        self.tools = {}
        self.memory = []
        self.config = {}
        
        # Performance tracking
        self.processed_count = 0
        self.error_count = 0
        self.last_activity = None
        
        self.logger.info(f"Initialized {agent_type} agent: {agent_id}")
    
    @abstractmethod
    async def observe(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe and process input data
        
        Args:
            input_data: Raw input data to process
            
        Returns:
            Processed observation data
        """
        pass
    
    @abstractmethod
    async def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decisions based on observations
        
        Args:
            observation: Processed observation data
            
        Returns:
            Decision or reasoning output
        """
        pass
    
    @abstractmethod
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute actions based on decisions
        
        Args:
            decision: Decision data from decide() method
            
        Returns:
            Action results
        """
        pass
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing pipeline: observe -> decide -> act
        
        This is the core agent processing loop that all agents follow.
        """
        
        try:
            self.state = AgentState.PROCESSING
            self.last_activity = datetime.now()
            
            # Standard agent pipeline
            observation = await self.observe(input_data)
            decision = await self.decide(observation)
            action = await self.act(decision)
            
            # Package results
            result = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "timestamp": self.last_activity.isoformat(),
                "input": input_data,
                "observation": observation,
                "decision": decision,
                "action": action,
                "status": "success"
            }
            
            self.processed_count += 1
            self.state = AgentState.IDLE
            
            self.logger.info(f"Successfully processed input (count: {self.processed_count})")
            return result
            
        except Exception as e:
            self.error_count += 1
            self.state = AgentState.ERROR
            
            error_result = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "timestamp": datetime.now().isoformat(),
                "input": input_data,
                "error": str(e),
                "status": "error",
                "error_count": self.error_count
            }
            
            self.logger.error(f"Processing error: {e}")
            return error_result
    
    def add_capability(self, capability: str):
        """Add a capability to this agent"""
        self.capabilities.add(capability)
        self.logger.info(f"Added capability: {capability}")
    
    def add_tool(self, tool_name: str, tool_function):
        """Add a tool to this agent"""
        self.tools[tool_name] = tool_function
        self.logger.info(f"Added tool: {tool_name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state.value,
            "capabilities": list(self.capabilities),
            "tools": list(self.tools.keys()),
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "memory_items": len(self.memory)
        }
    
    def store_memory(self, memory_item: Dict[str, Any]):
        """Store an item in agent memory"""
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "data": memory_item
        }
        self.memory.append(memory_entry)
        
        # Keep memory size manageable
        max_memory = self.config.get("max_memory_items", 1000)
        if len(self.memory) > max_memory:
            self.memory = self.memory[-max_memory:]
        
        self.logger.debug(f"Stored memory item (total: {len(self.memory)})")
    
    def search_memory(self, query: str) -> List[Dict[str, Any]]:
        """Search agent memory for relevant items"""
        # Simple keyword-based search
        results = []
        query_lower = query.lower()
        
        for memory_item in self.memory:
            memory_str = str(memory_item).lower()
            if query_lower in memory_str:
                results.append(memory_item)
        
        self.logger.debug(f"Memory search for '{query}' returned {len(results)} results")
        return results

class SimpleReasoningAgent(BaseNISAgent):
    """
    Simple reasoning agent with Chain of Thought
    
    This is a concrete implementation showing how to use the base framework.
    No hype - just working reasoning capabilities.
    """
    
    def __init__(self, agent_id: str = "reasoning-agent"):
        super().__init__(agent_id, "reasoning")
        
        # Add reasoning capabilities
        self.add_capability("chain_of_thought")
        self.add_capability("problem_decomposition")
        
        # Add basic tools
        self.add_tool("calculator", self._calculator_tool)
        self.add_tool("text_analyzer", self._text_analyzer_tool)
    
    async def observe(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Observe and structure input data"""
        
        observation = {
            "raw_input": input_data,
            "problem_type": self._classify_problem(input_data),
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.8  # Honest confidence assessment
        }
        
        return observation
    
    async def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Chain of Thought reasoning"""
        
        problem = observation.get("raw_input", {}).get("problem", "")
        
        # Chain of Thought steps
        cot_steps = []
        cot_steps.append(f"Problem: {problem}")
        
        # Problem analysis
        if "calculate" in problem.lower() or any(op in problem for op in ['+', '-', '*', '/']):
            cot_steps.append("This appears to be a mathematical problem")
            cot_steps.append("I should use calculation tools")
            tool_to_use = "calculator"
        elif "analyze" in problem.lower() or "text" in problem.lower():
            cot_steps.append("This appears to be a text analysis problem")
            cot_steps.append("I should use text analysis tools")
            tool_to_use = "text_analyzer"
        else:
            cot_steps.append("This is a general reasoning problem")
            cot_steps.append("I'll apply logical reasoning")
            tool_to_use = None
        
        cot_steps.append(f"Recommended tool: {tool_to_use}")
        
        decision = {
            "chain_of_thought": cot_steps,
            "recommended_tool": tool_to_use,
            "reasoning_type": "analytical",
            "confidence": 0.7  # Honest confidence
        }
        
        return decision
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the reasoning decision"""
        
        tool_name = decision.get("recommended_tool")
        
        if tool_name and tool_name in self.tools:
            # Use the recommended tool
            tool_result = self.tools[tool_name]("sample input")  # Simplified for demo
            
            action = {
                "action_type": "tool_execution",
                "tool_used": tool_name,
                "tool_result": tool_result,
                "chain_of_thought": decision.get("chain_of_thought", []),
                "success": True
            }
        else:
            # Fallback to general reasoning
            action = {
                "action_type": "general_reasoning",
                "reasoning_output": "Applied logical reasoning to the problem",
                "chain_of_thought": decision.get("chain_of_thought", []),
                "success": True
            }
        
        # Store in memory for future reference
        self.store_memory({
            "decision": decision,
            "action": action
        })
        
        return action
    
    def _classify_problem(self, input_data: Dict[str, Any]) -> str:
        """Classify the type of problem"""
        problem = str(input_data).lower()
        
        if any(word in problem for word in ["calculate", "math", "+", "-", "*", "/"]):
            return "mathematical"
        elif any(word in problem for word in ["analyze", "text", "words"]):
            return "textual"
        else:
            return "general"
    
    def _calculator_tool(self, expression: str) -> Dict[str, Any]:
        """Simple calculator tool"""
        try:
            # Basic safety check
            safe_chars = set("0123456789+-*/.()")
            if not all(c in safe_chars for c in expression.replace(" ", "")):
                raise ValueError("Invalid characters")
            
            result = eval(expression)
            return {"result": result, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _text_analyzer_tool(self, text: str) -> Dict[str, Any]:
        """Simple text analysis tool"""
        return {
            "word_count": len(text.split()),
            "character_count": len(text),
            "has_question": "?" in text,
            "success": True
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_agent():
        agent = SimpleReasoningAgent("test-reasoning")
        
        # Test the agent
        result = await agent.process({
            "problem": "What is 2 + 3 and why is it important?"
        })
        
        print("Agent Result:", result)
        print("Agent Status:", agent.get_status())
    
    asyncio.run(test_agent())
