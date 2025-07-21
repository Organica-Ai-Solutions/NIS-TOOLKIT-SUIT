#!/usr/bin/env python3
"""
Multi-Agent Coordination Example
Demonstrates enhanced NIS Agent Toolkit capabilities with all four agent types working together
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

# Import the enhanced agent templates
# Note: In production, these would be separate modules
from nis_agent_toolkit.core.base_agent import BaseNISAgent

class TaskCoordinator:
    """Coordinates multiple agents to accomplish complex tasks"""
    
    def __init__(self):
        self.agents = {}
        self.task_history = []
        self.shared_memory = {}
        
    def register_agent(self, agent_name: str, agent_instance: BaseNISAgent):
        """Register an agent with the coordinator"""
        self.agents[agent_name] = agent_instance
        print(f"✅ Registered {agent_instance.agent_type} agent: {agent_name}")
    
    async def coordinate_task(self, task_description: str) -> Dict[str, Any]:
        """Coordinate multiple agents to complete a complex task"""
        
        print(f"\n🎯 Starting coordinated task: {task_description}")
        print("=" * 60)
        
        # Break down the task using reasoning agent
        reasoning_result = await self._analyze_task(task_description)
        
        # Store task breakdown in memory
        memory_result = await self._store_task_info(task_description, reasoning_result)
        
        # Execute actions based on analysis
        action_results = await self._execute_task_actions(reasoning_result)
        
        # Analyze any visual content if applicable
        vision_results = await self._process_visual_content(task_description)
        
        # Compile final results
        final_result = {
            "task": task_description,
            "reasoning_analysis": reasoning_result,
            "memory_storage": memory_result,
            "actions_taken": action_results,
            "vision_analysis": vision_results,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        # Store completion in shared memory
        self.task_history.append(final_result)
        
        print("\n🎉 Task coordination complete!")
        print("=" * 60)
        
        return final_result
    
    async def _analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Use reasoning agent to analyze and break down the task"""
        if "reasoning_agent" not in self.agents:
            return {"error": "No reasoning agent available"}
        
        print("🧠 Analyzing task with reasoning agent...")
        
        reasoning_agent = self.agents["reasoning_agent"]
        result = await reasoning_agent.process({
            "problem": f"Analyze this task and break it into steps: {task_description}"
        })
        
        print(f"   ✓ Reasoning complete - confidence: {result['action'].get('confidence', 0)}")
        return result
    
    async def _store_task_info(self, task: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Use memory agent to store task information"""
        if "memory_agent" not in self.agents:
            return {"error": "No memory agent available"}
        
        print("💾 Storing task information in memory...")
        
        memory_agent = self.agents["memory_agent"]
        result = await memory_agent.process({
            "operation": "store",
            "content": f"Task: {task}. Analysis: {json.dumps(analysis['action'], indent=2)}",
            "category": "tasks",
            "importance": 0.8
        })
        
        print(f"   ✓ Memory storage complete - items stored: {result['action']['memory_result'].get('items_processed', 0)}")
        return result
    
    async def _execute_task_actions(self, reasoning_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use action agent to execute identified actions"""
        if "action_agent" not in self.agents:
            return [{"error": "No action agent available"}]
        
        print("⚡ Executing task actions...")
        
        action_agent = self.agents["action_agent"]
        
        # Extract potential actions from reasoning chain
        reasoning_chain = reasoning_result.get("action", {}).get("reasoning_chain", [])
        
        actions_to_execute = []
        for step in reasoning_chain:
            if any(keyword in step.lower() for keyword in ["execute", "run", "check", "list"]):
                if "list" in step.lower():
                    actions_to_execute.append({"action_type": "command", "command": "ls -la"})
                elif "check" in step.lower():
                    actions_to_execute.append({"action_type": "command", "command": "pwd"})
        
        # Execute actions
        action_results = []
        for action in actions_to_execute:
            result = await action_agent.process(action)
            action_results.append(result)
            print(f"   ✓ Action executed: {action['command']} - success: {result.get('action', {}).get('success', False)}")
        
        return action_results
    
    async def _process_visual_content(self, task_description: str) -> Dict[str, Any]:
        """Use vision agent if task involves visual content"""
        if "vision_agent" not in self.agents:
            return {"info": "No vision agent available"}
        
        # Check if task involves visual content
        visual_keywords = ["image", "picture", "photo", "visual", "analyze photo", "look at"]
        if not any(keyword in task_description.lower() for keyword in visual_keywords):
            return {"info": "No visual content detected"}
        
        print("👁️ Processing visual content...")
        
        vision_agent = self.agents["vision_agent"]
        
        # Simulate processing a task-related image
        result = await vision_agent.process({
            "image_path": "task_related_image.jpg",
            "analysis_type": "general"
        })
        
        print(f"   ✓ Vision analysis complete - confidence: {result['action'].get('confidence', 0)}")
        return result

# Example agent implementations (simplified for demonstration)
class EnhancedReasoningAgent(BaseNISAgent):
    """Enhanced reasoning agent with better Chain of Thought"""
    
    def __init__(self):
        super().__init__("reasoning-agent", "reasoning")
        self.add_capability("chain_of_thought")
        self.add_capability("task_decomposition")
    
    async def observe(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        problem = input_data.get("problem", "")
        return {
            "problem": problem,
            "complexity": "high" if len(problem) > 50 else "medium",
            "keywords": problem.lower().split()[:5],
            "confidence": 0.85
        }
    
    async def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        problem = observation["problem"]
        
        reasoning_steps = [
            "📋 Task decomposition starting...",
            f"🎯 Primary objective: {problem}",
            "🔍 Identifying sub-tasks...",
            "⚡ Planning execution steps...",
            "✅ Ready for action coordination"
        ]
        
        return {
            "approach": "task_decomposition",
            "reasoning_chain": reasoning_steps,
            "sub_tasks": ["analyze", "plan", "execute", "verify"],
            "confidence": 0.9
        }
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action_type": "reasoning_complete",
            "reasoning_chain": decision["reasoning_chain"],
            "recommended_actions": decision["sub_tasks"],
            "confidence": decision["confidence"],
            "success": True
        }

class EnhancedMemoryAgent(BaseNISAgent):
    """Enhanced memory agent with relationship tracking"""
    
    def __init__(self):
        super().__init__("memory-agent", "memory")
        self.add_capability("intelligent_storage")
        self.memory_store = {}
    
    async def observe(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "operation": input_data.get("operation", "store"),
            "content": input_data.get("content", ""),
            "category": input_data.get("category", "general"),
            "confidence": 0.9
        }
    
    async def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "approach": observation["operation"],
            "storage_strategy": "intelligent_categorization",
            "confidence": 0.88
        }
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action_type": "memory_operation_complete",
            "memory_result": {
                "operation": decision["approach"],
                "items_processed": 1,
                "success": True
            },
            "success": True
        }

class EnhancedActionAgent(BaseNISAgent):
    """Enhanced action agent with safety controls"""
    
    def __init__(self):
        super().__init__("action-agent", "action")
        self.add_capability("safe_command_execution")
    
    async def observe(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action_type": input_data.get("action_type", "command"),
            "command": input_data.get("command", ""),
            "safety_approved": True,
            "confidence": 0.8
        }
    
    async def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "approach": "safe_execution",
            "execution_plan": ["validate", "execute", "report"],
            "confidence": 0.85
        }
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action_type": "command_executed",
            "success": True,
            "confidence": decision["confidence"]
        }

class EnhancedVisionAgent(BaseNISAgent):
    """Enhanced vision agent with content analysis"""
    
    def __init__(self):
        super().__init__("vision-agent", "vision")
        self.add_capability("intelligent_image_analysis")
    
    async def observe(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_path": input_data.get("image_path", ""),
            "analysis_type": input_data.get("analysis_type", "general"),
            "image_valid": True,
            "confidence": 0.87
        }
    
    async def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "approach": observation["analysis_type"],
            "processing_strategy": "multi_layer_analysis",
            "confidence": 0.9
        }
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action_type": "vision_analysis_complete",
            "vision_results": {
                "objects_detected": ["computer", "desk", "documents"],
                "scene_type": "office_workspace",
                "confidence": 0.92
            },
            "success": True,
            "confidence": decision["confidence"]
        }

async def demonstrate_coordination():
    """Demonstrate multi-agent coordination capabilities"""
    
    print("🚀 NIS Agent Toolkit - Multi-Agent Coordination Demo")
    print("=" * 60)
    
    # Initialize the coordinator
    coordinator = TaskCoordinator()
    
    # Create and register enhanced agents
    reasoning_agent = EnhancedReasoningAgent()
    memory_agent = EnhancedMemoryAgent()
    action_agent = EnhancedActionAgent()
    vision_agent = EnhancedVisionAgent()
    
    coordinator.register_agent("reasoning_agent", reasoning_agent)
    coordinator.register_agent("memory_agent", memory_agent)
    coordinator.register_agent("action_agent", action_agent)
    coordinator.register_agent("vision_agent", vision_agent)
    
    # Test scenarios
    test_scenarios = [
        "Organize project files and analyze the current workspace structure",
        "Review system status and create a summary report",
        "Process uploaded images and extract key information for documentation"
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔄 Scenario {i}: {scenario}")
        result = await coordinator.coordinate_task(scenario)
        
        # Show results summary
        print(f"\n📊 Results Summary:")
        print(f"   • Reasoning steps: {len(result['reasoning_analysis']['action'].get('reasoning_chain', []))}")
        print(f"   • Memory operations: {result['memory_storage']['action']['memory_result']['items_processed']}")
        print(f"   • Actions executed: {len(result['actions_taken'])}")
        print(f"   • Overall success: {result['success']}")
        
        await asyncio.sleep(1)  # Brief pause between scenarios
    
    print(f"\n🎉 Multi-Agent Coordination Demo Complete!")
    print(f"📈 Total scenarios completed: {len(test_scenarios)}")
    print(f"🧠 Agents demonstrated: reasoning, memory, action, vision")
    print(f"⚡ All systems working together seamlessly!")

if __name__ == "__main__":
    asyncio.run(demonstrate_coordination()) 