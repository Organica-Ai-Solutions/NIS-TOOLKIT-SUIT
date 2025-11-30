#!/usr/bin/env python3
"""
NIS Core Toolkit - Component Creation
Creates agents, protocols, and other system components
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()

def create_agent(agent_name: str, agent_type: str = "reasoning"):
    """Create a new agent using NIS Agent Toolkit"""
    
    # Check if we're in a NIS project
    if not Path("config/project.yaml").exists():
        console.print("❌ Not in a NIS project directory", style="red")
        console.print("Run 'nis init' to create a new project first")
        return False
    
    # Check if agents directory exists
    agents_dir = Path("agents")
    if not agents_dir.exists():
        agents_dir.mkdir()
    
    agent_dir = agents_dir / agent_name
    if agent_dir.exists():
        console.print(f"❌ Agent '{agent_name}' already exists", style="red")
        return False
    
