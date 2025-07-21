#!/usr/bin/env python3
"""
NIS Agent Toolkit - Main CLI
Agent-level development tools for NIS Protocol
"""

import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
@click.version_option(version="1.0.0")
def main():
    """NIS Agent Toolkit - Agent-level developer toolkit"""
    pass

@main.command()
def status():
    """Show agent toolkit status"""
    console.print("í´– NIS Agent Toolkit v1.0.0", style="bold blue")
    console.print("Agent-level developer toolkit for NIS Protocol\n")
    
    table = Table(title="Available Commands")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")
    table.add_column("Status", style="green")
    
    table.add_row("nis-agent create", "Create new agent from template", "âœ… Working")
    table.add_row("nis-agent test", "Test agent functionality", "âœ… Working")
    table.add_row("nis-agent simulate", "Run agent simulation", "âœ… Working")
    table.add_row("nis-agent eval", "Evaluate agent performance", "ï¿½ï¿½ Planned")
    
    console.print(table)

@main.command()
@click.argument("agent_type", type=click.Choice(["reasoning", "vision", "memory", "action"]))
@click.argument("agent_name")
def create(agent_type, agent_name):
    """Create a new agent from template"""
    from .create import create_agent
    create_agent(agent_name, agent_type)

@main.command()
@click.argument("agent_name")
def test(agent_name):
    """Test an agent"""
    from .test import test_agent
    test_agent(agent_name)

@main.command()
@click.argument("agent_name")
def simulate(agent_name):
    """Simulate agent behavior"""
    from .simulate import simulate_agent
    simulate_agent(agent_name)

if __name__ == "__main__":
    main()
