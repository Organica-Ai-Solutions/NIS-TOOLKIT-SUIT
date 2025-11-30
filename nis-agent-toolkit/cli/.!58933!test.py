#!/usr/bin/env python3
"""
NIS Agent Toolkit - Agent Testing Framework
Test agent functionality with real test cases
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Dict, Any, List

console = Console()

class AgentTester:
    """
    Agent testing framework with honest evaluation
    No hype - just practical testing of agent capabilities
    """
    
    def __init__(self, agent_path: Path):
        self.agent_path = agent_path
        self.agent = None
        self.test_results = []
    
    def load_agent(self) -> bool:
        """Load agent from file"""
        
        agent_files = list(self.agent_path.glob("*.py"))
        if not agent_files:
            console.print("❌ No Python agent file found", style="red")
            return False
        
        agent_file = agent_files[0]  # Use first Python file
        
        try:
            # Load the agent module
            spec = importlib.util.spec_from_file_location("agent_module", agent_file)
            agent_module = importlib.util.module_from_spec(spec)
            sys.modules["agent_module"] = agent_module
            spec.loader.exec_module(agent_module)
            
            # Find agent class (assumes class name matches file pattern)
            agent_classes = [obj for name, obj in vars(agent_module).items() 
                           if isinstance(obj, type) and name.endswith('Agent') or name.endswith('Agent')]
            
            if not agent_classes:
                console.print("❌ No agent class found in file", style="red")
                return False
            
            # Instantiate the agent
            agent_class = agent_classes[0]
            self.agent = agent_class()
            
            console.print(f"✅ Loaded agent: {agent_class.__name__}", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Error loading agent: {e}", style="red")
            return False
    
    async def run_basic_tests(self) -> Dict[str, Any]:
        """Run basic functionality tests"""
        
