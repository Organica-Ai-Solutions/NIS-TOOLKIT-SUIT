#!/usr/bin/env python3
"""
NIS Agent Toolkit - Agent Simulation Framework
Simulate agent behavior in controlled environments
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, TaskID
import importlib.util
import sys

console = Console()

@dataclass
class SimulationScenario:
    """Simulation scenario configuration"""
    name: str
    description: str
    inputs: List[Dict[str, Any]]
    expected_outputs: Optional[List[Dict[str, Any]]] = None
    timeout_seconds: int = 30
    environment_config: Optional[Dict[str, Any]] = None

class AgentSimulator:
    """
    Agent simulation framework with honest evaluation
    Tests agents in controlled scenarios - no hype, just practical testing
    """
    
    def __init__(self, agent_path: Path):
        self.agent_path = agent_path
        self.agent = None
        self.simulation_results = []
        self.scenarios = []
        
    def load_agent(self) -> bool:
        """Load agent from file"""
        
        agent_files = list(self.agent_path.glob("*.py"))
        if not agent_files:
            console.print("❌ No Python agent file found", style="red")
            return False
        
        agent_file = agent_files[0]
        
        try:
            # Load the agent module
            spec = importlib.util.spec_from_file_location("agent_module", agent_file)
            agent_module = importlib.util.module_from_spec(spec)
            sys.modules["agent_module"] = agent_module
            spec.loader.exec_module(agent_module)
            
            # Find agent class
            agent_classes = [obj for name, obj in vars(agent_module).items() 
                           if isinstance(obj, type) and hasattr(obj, 'process')]
            
            if not agent_classes:
                console.print("❌ No agent class found", style="red")
                return False
            
            # Instantiate the agent
            agent_class = agent_classes[0]
            self.agent = agent_class()
            
            console.print(f"✅ Loaded agent: {agent_class.__name__}", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Error loading agent: {e}", style="red")
            return False
    
    def load_scenarios(self, scenario_file: Optional[Path] = None) -> List[SimulationScenario]:
        """Load simulation scenarios"""
        
        if scenario_file and scenario_file.exists():
            # Load from file
            try:
                with open(scenario_file, 'r') as f:
                    scenarios_data = json.load(f)
                
                scenarios = []
                for scenario_data in scenarios_data:
                    scenario = SimulationScenario(**scenario_data)
                    scenarios.append(scenario)
                
                console.print(f"✅ Loaded {len(scenarios)} scenarios from file", style="green")
                return scenarios
                
            except Exception as e:
                console.print(f"❌ Error loading scenarios: {e}", style="red")
        
        # Default scenarios
        return self._get_default_scenarios()
    
    def _get_default_scenarios(self) -> List[SimulationScenario]:
        """Get default simulation scenarios"""
        
        scenarios = [
            SimulationScenario(
                name="Basic Processing",
                description="Test basic agent processing with simple input",
                inputs=[
                    {"problem": "What is 2 + 2?"},
                    {"problem": "Explain what the sky is blue"},
                    {"problem": "Process this simple request"}
                ]
            ),
            SimulationScenario(
                name="Mathematical Reasoning",
                description="Test mathematical problem solving",
                inputs=[
                    {"problem": "Calculate 15 * 23 + 7"},
                    {"problem": "What is the square root of 144?"},
                    {"problem": "Solve for x: 2x + 5 = 13"}
                ]
            ),
            SimulationScenario(
                name="Text Analysis",
                description="Test text processing capabilities",
                inputs=[
                    {"problem": "Analyze this text: 'The quick brown fox jumps over the lazy dog'"},
                    {"problem": "Count the words in this sentence"},
                    {"problem": "Extract keywords from this content"}
                ]
            ),
            SimulationScenario(
                name="Error Handling",
                description="Test agent behavior with invalid inputs",
                inputs=[
                    {"invalid": "input"},
                    {"problem": None},
                    {}
                ]
            ),
            SimulationScenario(
                name="Complex Reasoning",
                description="Test multi-step reasoning capabilities",
                inputs=[
                    {"problem": "Plan a route from point A to point B considering traffic"},
                    {"problem": "Analyze the pros and cons of renewable energy"},
                    {"problem": "Explain the water cycle step by step"}
                ]
            )
        ]
        
        return scenarios
    
    async def run_simulation(self, scenarios: List[SimulationScenario]) -> Dict[str, Any]:
        """Run complete simulation suite"""
        
        console.print("🎮 Starting agent simulation...", style="bold blue")
        
        simulation_start = time.time()
        results = []
        
        with Progress() as progress:
            task = progress.add_task("Simulating...", total=len(scenarios))
            
            for scenario in scenarios:
                scenario_result = await self._run_scenario(scenario)
                results.append(scenario_result)
                progress.update(task, advance=1)
        
        simulation_end = time.time()
        simulation_duration = simulation_end - simulation_start
        
        # Calculate overall metrics
        total_inputs = sum(len(scenario.inputs) for scenario in scenarios)
        successful_runs = sum(r["successful_runs"] for r in results)
        failed_runs = sum(r["failed_runs"] for r in results)
        
        overall_result = {
            "simulation_duration": simulation_duration,
            "scenarios_run": len(scenarios),
            "total_inputs": total_inputs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": successful_runs / total_inputs if total_inputs > 0 else 0,
            "scenario_results": results
        }
        
        return overall_result
    
    async def _run_scenario(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """Run a single simulation scenario"""
        
        console.print(f"🔄 Running scenario: {scenario.name}", style="blue")
        
        scenario_start = time.time()
        successful_runs = 0
        failed_runs = 0
        input_results = []
        
        for i, input_data in enumerate(scenario.inputs):
            try:
                # Run with timeout
                result = await asyncio.wait_for(
                    self.agent.process(input_data),
                    timeout=scenario.timeout_seconds
                )
                
                # Check if successful
                success = result.get("status") != "error"
                if success:
                    successful_runs += 1
                else:
                    failed_runs += 1
                
                input_results.append({
                    "input_index": i,
                    "input": input_data,
                    "result": result,
                    "success": success,
                    "processing_time": result.get("processing_time", "unknown")
                })
                
            except asyncio.TimeoutError:
                failed_runs += 1
                input_results.append({
                    "input_index": i,
                    "input": input_data,
                    "result": {"error": "Timeout"},
                    "success": False,
                    "processing_time": "timeout"
                })
                
            except Exception as e:
                failed_runs += 1
                input_results.append({
                    "input_index": i,
                    "input": input_data,
                    "result": {"error": str(e)},
                    "success": False,
                    "processing_time": "error"
                })
        
        scenario_end = time.time()
        scenario_duration = scenario_end - scenario_start
        
        return {
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "duration": scenario_duration,
            "total_inputs": len(scenario.inputs),
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": successful_runs / len(scenario.inputs) if scenario.inputs else 0,
            "input_results": input_results
        }
    
    def display_simulation_results(self, results: Dict[str, Any]):
        """Display simulation results"""
        
        console.print("\n" + "="*60)
        console.print("🎮 AGENT SIMULATION REPORT", style="bold blue")
        console.print("="*60)
        
        # Overall summary
        console.print(f"⏱️  Total Duration: {results['simulation_duration']:.2f} seconds")
        console.print(f"🎯 Scenarios Run: {results['scenarios_run']}")
        console.print(f"📊 Total Inputs: {results['total_inputs']}")
        console.print(f"✅ Successful: {results['successful_runs']}")
        console.print(f"❌ Failed: {results['failed_runs']}")
        console.print(f"📈 Success Rate: {results['success_rate']:.1%}")
        
        # Scenario breakdown
        console.print("\n📋 SCENARIO BREAKDOWN:", style="bold")
        
        table = Table(title="Scenario Results")
        table.add_column("Scenario", style="cyan")
        table.add_column("Success Rate", style="green")
        table.add_column("Duration", style="magenta")
        table.add_column("Status", style="yellow")
        
        for scenario_result in results["scenario_results"]:
            success_rate = scenario_result["success_rate"]
            duration = f"{scenario_result['duration']:.2f}s"
            
            if success_rate >= 0.8:
                status = "✅ PASS"
            elif success_rate >= 0.5:
                status = "⚠️  WARN"
            else:
                status = "❌ FAIL"
            
            table.add_row(
                scenario_result["scenario_name"],
                f"{success_rate:.1%}",
                duration,
                status
            )
        
        console.print(table)
        
        # Detailed results for failed scenarios
        failed_scenarios = [r for r in results["scenario_results"] if r["success_rate"] < 0.8]
        
        if failed_scenarios:
            console.print("\n🔍 FAILED SCENARIO DETAILS:", style="bold red")
            
            for scenario_result in failed_scenarios:
                console.print(f"\n❌ {scenario_result['scenario_name']}:")
                
                failed_inputs = [r for r in scenario_result["input_results"] if not r["success"]]
                for failed_input in failed_inputs:
                    console.print(f"   Input {failed_input['input_index']}: {failed_input['result'].get('error', 'Unknown error')}")
    
    def save_results(self, results: Dict[str, Any], output_file: Path):
        """Save simulation results to file"""
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"💾 Results saved to: {output_file}", style="green")
            
        except Exception as e:
            console.print(f"❌ Error saving results: {e}", style="red")

class InteractiveSimulator:
    """Interactive simulation mode for real-time testing"""
    
    def __init__(self, agent):
        self.agent = agent
        self.session_history = []
    
    async def run_interactive(self):
        """Run interactive simulation session"""
        
        console.print("🎮 Interactive Agent Simulation Mode", style="bold blue")
        console.print("Type 'exit' to quit, 'help' for commands\n")
        
        while True:
            try:
                # Get user input
                user_input = console.input("📝 Enter test input: ")
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                # Process input
                console.print("🔄 Processing...", style="blue")
                
                start_time = time.time()
                try:
                    result = await self.agent.process({"problem": user_input})
                    processing_time = time.time() - start_time
                    
                    # Display result
                    console.print("✅ Result:", style="green")
                    console.print(f"   Status: {result.get('status', 'unknown')}")
                    console.print(f"   Processing Time: {processing_time:.2f}s")
                    
                    if "action" in result and "final_answer" in result["action"]:
                        console.print(f"   Answer: {result['action']['final_answer']}")
                    
                    if "action" in result and "reasoning_chain" in result["action"]:
                        console.print("   Reasoning:")
                        for step in result["action"]["reasoning_chain"][-3:]:  # Show last 3 steps
                            console.print(f"     • {step}")
                    
                    # Save to history
                    self.session_history.append({
                        "input": user_input,
                        "result": result,
                        "processing_time": processing_time
                    })
                    
                except Exception as e:
                    console.print(f"❌ Error: {e}", style="red")
                
                console.print()  # Empty line for readability
                
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!", style="blue")
                break
            except Exception as e:
                console.print(f"❌ Session error: {e}", style="red")
    
    def _show_help(self):
        """Show help commands"""
        console.print("📚 Available Commands:", style="bold")
        console.print("   exit     - Exit interactive mode")
        console.print("   help     - Show this help")
        console.print("   history  - Show session history")
        console.print("   Or enter any text to test the agent\n")
    
    def _show_history(self):
        """Show session history"""
        if not self.session_history:
            console.print("📝 No history available", style="yellow")
            return
        
        console.print(f"📜 Session History ({len(self.session_history)} items):", style="bold")
        for i, entry in enumerate(self.session_history[-5:], 1):  # Show last 5
            console.print(f"   {i}. {entry['input'][:50]}...")
            console.print(f"      → {entry['result'].get('status', 'unknown')} ({entry['processing_time']:.2f}s)")
        console.print()

def simulate_agent(agent_name: str, scenario_file: str = None, interactive: bool = False):
    """Main simulation function"""
    
    agent_path = Path(agent_name)
    
    if not agent_path.exists():
        console.print(f"❌ Agent directory '{agent_name}' not found", style="red")
        return False
    
    # Load agent
    simulator = AgentSimulator(agent_path)
    if not simulator.load_agent():
        return False
    
    # Interactive mode
    if interactive:
        interactive_sim = InteractiveSimulator(simulator.agent)
        asyncio.run(interactive_sim.run_interactive())
        return True
    
    # Batch simulation mode
    async def run_batch():
        # Load scenarios
        scenario_file_path = Path(scenario_file) if scenario_file else None
        scenarios = simulator.load_scenarios(scenario_file_path)
        
        # Run simulation
        results = await simulator.run_simulation(scenarios)
        
        # Display results
        simulator.display_simulation_results(results)
        
        # Save results
        output_file = Path(f"{agent_name}_simulation_results.json")
        simulator.save_results(results, output_file)
        
        return results["success_rate"] > 0.7  # Consider successful if > 70%
    
    return asyncio.run(run_batch())

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NIS Agent Simulation")
    parser.add_argument("agent_name", help="Agent name/directory to simulate")
    parser.add_argument("--scenario-file", help="JSON file with simulation scenarios")
    parser.add_argument("--interactive", action="store_true", help="Run interactive simulation")
    
    args = parser.parse_args()
    
    success = simulate_agent(args.agent_name, args.scenario_file, args.interactive)
    
    if success:
        console.print("✅ Simulation completed successfully!", style="bold green")
    else:
        console.print("❌ Simulation failed!", style="bold red")

if __name__ == "__main__":
    main() 