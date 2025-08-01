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
            console.print("âŒ No Python agent file found", style="red")
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
                console.print("âŒ No agent class found in file", style="red")
                return False
            
            # Instantiate the agent
            agent_class = agent_classes[0]
            self.agent = agent_class()
            
            console.print(f"âœ… Loaded agent: {agent_class.__name__}", style="green")
            return True
            
        except Exception as e:
            console.print(f"âŒ Error loading agent: {e}", style="red")
            return False
    
    async def run_basic_tests(self) -> Dict[str, Any]:
        """Run basic functionality tests"""
        
        console.print("í·ª Running basic agent tests...", style="bold blue")
        
        tests = [
            {
                "name": "Interface Test",
                "test": self._test_interface,
                "required": True
            },
            {
                "name": "Processing Test",
                "test": self._test_processing,
                "required": True
            },
            {
                "name": "Error Handling Test",
                "test": self._test_error_handling,
                "required": False
            },
            {
                "name": "Memory Test",
                "test": self._test_memory,
                "required": False
            }
        ]
        
        results = []
        for test_info in tests:
            try:
                result = await test_info["test"]()
                results.append({
                    "name": test_info["name"],
                    "passed": result.get("success", False),
                    "details": result,
                    "required": test_info["required"]
                })
            except Exception as e:
                results.append({
                    "name": test_info["name"],
                    "passed": False,
                    "details": {"error": str(e)},
                    "required": test_info["required"]
                })
        
        return {"tests": results, "agent_id": getattr(self.agent, 'agent_id', 'unknown')}
    
    async def _test_interface(self) -> Dict[str, Any]:
        """Test agent interface compliance"""
        
        required_methods = ["process", "observe", "decide", "act"]
        missing_methods = []
        
        for method in required_methods:
            if not hasattr(self.agent, method):
                missing_methods.append(method)
        
        return {
            "success": len(missing_methods) == 0,
            "required_methods": required_methods,
            "missing_methods": missing_methods,
            "has_agent_id": hasattr(self.agent, 'agent_id'),
            "has_capabilities": hasattr(self.agent, 'capabilities')
        }
    
    async def _test_processing(self) -> Dict[str, Any]:
        """Test basic processing functionality"""
        
        test_input = {
            "problem": "Test input for agent processing",
            "test_mode": True
        }
        
        try:
            result = await self.agent.process(test_input)
            
            return {
                "success": True,
                "input": test_input,
                "output": result,
                "has_status": "status" in result,
                "processing_time": "Quick"  # Simplified timing
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "input": test_input
            }
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid input"""
        
        invalid_inputs = [
            None,
            {"invalid": "input structure"},
            {"problem": None}
        ]
        
        error_handled = 0
        for invalid_input in invalid_inputs:
            try:
                result = await self.agent.process(invalid_input)
                if result.get("status") == "error":
                    error_handled += 1
            except Exception:
                # Agent should handle errors gracefully, not crash
                pass
        
        return {
            "success": error_handled > 0,
            "tests_run": len(invalid_inputs),
            "errors_handled": error_handled
        }
    
    async def _test_memory(self) -> Dict[str, Any]:
        """Test memory functionality if available"""
        
        if not hasattr(self.agent, 'memory'):
            return {"success": True, "note": "No memory system found"}
        
        initial_memory_size = len(self.agent.memory)
        
        # Process something to potentially add to memory
        await self.agent.process({"problem": "Memory test input"})
        
        final_memory_size = len(self.agent.memory)
        
        return {
            "success": True,
            "initial_memory_size": initial_memory_size,
            "final_memory_size": final_memory_size,
            "memory_changed": final_memory_size != initial_memory_size
        }
    
    def display_results(self, results: Dict[str, Any]):
        """Display test results in a nice format"""
        
        console.print(f"\ní³Š Test Results for Agent: {results['agent_id']}", style="bold")
        
        table = Table(title="Test Summary")
        table.add_column("Test Name", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Required", style="yellow")
        table.add_column("Details", style="green")
        
        passed_tests = 0
        required_tests_passed = 0
        total_required = 0
        
        for test in results["tests"]:
            status = "âœ… PASS" if test["passed"] else "âŒ FAIL"
            required = "Yes" if test["required"] else "No"
            
            # Summarize details
            details = test["details"]
            if test["passed"]:
                detail_str = "All checks passed"
            else:
                detail_str = details.get("error", "Check failed")
            
            table.add_row(test["name"], status, required, detail_str)
            
            if test["passed"]:
                passed_tests += 1
                if test["required"]:
                    required_tests_passed += 1
            
            if test["required"]:
                total_required += 1
        
        console.print(table)
        
        # Summary
        console.print(f"\ní³ˆ Summary:", style="bold")
        console.print(f"Total tests: {len(results['tests'])}")
        console.print(f"Passed: {passed_tests}")
        console.print(f"Required tests passed: {required_tests_passed}/{total_required}")
        
        if required_tests_passed == total_required:
            console.print("í¿¢ Agent is ready for deployment!", style="bold green")
        else:
            console.print("í´´ Agent needs fixes before deployment", style="bold red")

def test_agent(agent_name: str):
    """Main function to test an agent"""
    
    agent_path = Path(agent_name)
    
    if not agent_path.exists():
        console.print(f"âŒ Agent directory '{agent_name}' not found", style="red")
        return
    
    tester = AgentTester(agent_path)
    
    if not tester.load_agent():
        return
    
    # Run tests
    async def run_tests():
        results = await tester.run_basic_tests()
        tester.display_results(results)
    
    asyncio.run(run_tests())

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_agent(sys.argv[1])
    else:
        console.print("Usage: python test.py <agent_name>")
