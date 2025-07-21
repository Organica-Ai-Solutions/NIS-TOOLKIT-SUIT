#!/usr/bin/env python3
"""
NIS Core Toolkit - Project Validation
Validate NIS projects for protocol compliance and structure
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

class NISValidator:
    """
    NIS Protocol compliance validator
    Honest validation - no hype, just practical checks
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(".")
        self.validation_results = []
        self.errors = []
        self.warnings = []
    
    def validate_project(self) -> Dict[str, Any]:
        """Run complete project validation"""
        
        console.print("🔍 Validating NIS project structure...", style="bold blue")
        
        # Core validation checks
        checks = [
            ("Project Structure", self._validate_structure),
            ("Configuration", self._validate_config),
            ("Agent Compliance", self._validate_agents),
            ("Protocol Adherence", self._validate_protocol),
            ("Dependencies", self._validate_dependencies)
        ]
        
        results = {}
        for check_name, check_func in checks:
            try:
                result = check_func()
                results[check_name] = result
                
                if result.get("status") == "pass":
                    console.print(f"✅ {check_name}: PASS", style="green")
                elif result.get("status") == "warning":
                    console.print(f"⚠️  {check_name}: WARNING", style="yellow")
                else:
                    console.print(f"❌ {check_name}: FAIL", style="red")
                    
            except Exception as e:
                results[check_name] = {"status": "error", "error": str(e)}
                console.print(f"❌ {check_name}: ERROR - {e}", style="red")
        
        return {
            "overall_status": self._determine_overall_status(results),
            "results": results,
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def _validate_structure(self) -> Dict[str, Any]:
        """Validate project directory structure"""
        
        required_dirs = ["agents", "config", "logs"]
        optional_dirs = ["tests", "data", "docs"]
        required_files = ["main.py", "requirements.txt"]
        
        missing_dirs = []
        missing_files = []
        
        # Check required directories
        for dir_name in required_dirs:
            if not (self.project_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        # Check required files
        for file_name in required_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)
        
        # Check optional directories (warnings only)
        missing_optional = []
        for dir_name in optional_dirs:
            if not (self.project_root / dir_name).exists():
                missing_optional.append(dir_name)
        
        status = "pass"
        if missing_dirs or missing_files:
            status = "fail"
        elif missing_optional:
            status = "warning"
        
        return {
            "status": status,
            "required_dirs": required_dirs,
            "missing_dirs": missing_dirs,
            "required_files": required_files,
            "missing_files": missing_files,
            "missing_optional": missing_optional
        }
    
    def _validate_config(self) -> Dict[str, Any]:
        """Validate project configuration"""
        
        config_file = self.project_root / "config" / "project.yaml"
        
        if not config_file.exists():
            return {"status": "fail", "error": "No project.yaml found"}
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            required_keys = ["project", "nis"]
            missing_keys = []
            
            for key in required_keys:
                if key not in config:
                    missing_keys.append(key)
            
            # Check NIS protocol version
            protocol_version = config.get("nis", {}).get("protocol_version")
            if not protocol_version:
                missing_keys.append("nis.protocol_version")
            
            status = "pass" if not missing_keys else "fail"
            
            return {
                "status": status,
                "config_file": str(config_file),
                "required_keys": required_keys,
                "missing_keys": missing_keys,
                "protocol_version": protocol_version,
                "config_valid": True
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "error": f"Config parsing error: {e}",
                "config_file": str(config_file)
            }
    
    def _validate_agents(self) -> Dict[str, Any]:
        """Validate agent implementations"""
        
        agents_dir = self.project_root / "agents"
        
        if not agents_dir.exists():
            return {"status": "fail", "error": "No agents directory found"}
        
        agent_dirs = [d for d in agents_dir.iterdir() if d.is_dir()]
        
        if not agent_dirs:
            return {"status": "warning", "message": "No agents found"}
        
        agent_results = []
        for agent_dir in agent_dirs:
            agent_result = self._validate_single_agent(agent_dir)
            agent_results.append(agent_result)
        
        # Overall status
        failed_agents = [r for r in agent_results if r.get("status") == "fail"]
        warning_agents = [r for r in agent_results if r.get("status") == "warning"]
        
        if failed_agents:
            status = "fail"
        elif warning_agents:
            status = "warning"
        else:
            status = "pass"
        
        return {
            "status": status,
            "agents_found": len(agent_dirs),
            "agents_passed": len([r for r in agent_results if r.get("status") == "pass"]),
            "agents_failed": len(failed_agents),
            "agent_results": agent_results
        }
    
    def _validate_single_agent(self, agent_dir: Path) -> Dict[str, Any]:
        """Validate a single agent"""
        
        agent_name = agent_dir.name
        agent_files = list(agent_dir.glob("*.py"))
        
        if not agent_files:
            return {
                "name": agent_name,
                "status": "fail",
                "error": "No Python files found"
            }
        
        # Basic checks
        has_main_file = len(agent_files) > 0
        has_config = (agent_dir / "config.yaml").exists()
        
        # Try to validate Python syntax
        syntax_valid = True
        syntax_errors = []
        
        for py_file in agent_files:
            try:
                with open(py_file, 'r') as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                syntax_valid = False
                syntax_errors.append(f"{py_file.name}: {e}")
        
        status = "pass"
        if not syntax_valid:
            status = "fail"
        elif not has_config:
            status = "warning"
        
        return {
            "name": agent_name,
            "status": status,
            "has_main_file": has_main_file,
            "has_config": has_config,
            "syntax_valid": syntax_valid,
            "syntax_errors": syntax_errors,
            "files_found": len(agent_files)
        }
    
    def _validate_protocol(self) -> Dict[str, Any]:
        """Validate NIS protocol compliance"""
        
        # Check for protocol markers
        protocol_markers = [
            "BaseNISAgent",
            "observe",
            "decide", 
            "act",
            "process"
        ]
        
        found_markers = []
        agent_files = list(self.project_root.glob("agents/*/*.py"))
        
        for agent_file in agent_files:
            try:
                with open(agent_file, 'r') as f:
                    content = f.read()
                    for marker in protocol_markers:
                        if marker in content:
                            found_markers.append(marker)
            except Exception:
                pass
        
        protocol_compliance = len(set(found_markers)) / len(protocol_markers)
        
        status = "pass" if protocol_compliance >= 0.8 else "warning" if protocol_compliance >= 0.5 else "fail"
        
        return {
            "status": status,
            "protocol_compliance": protocol_compliance,
            "found_markers": list(set(found_markers)),
            "missing_markers": list(set(protocol_markers) - set(found_markers))
        }
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate project dependencies"""
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            return {"status": "fail", "error": "No requirements.txt found"}
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = f.read().strip().split('\n')
            
            # Check for core NIS dependencies
            core_deps = ["nis-core-toolkit", "nis-agent-toolkit"]
            missing_deps = []
            
            for dep in core_deps:
                if not any(dep in req for req in requirements):
                    missing_deps.append(dep)
            
            status = "pass" if not missing_deps else "warning"
            
            return {
                "status": status,
                "requirements_count": len(requirements),
                "missing_core_deps": missing_deps,
                "has_requirements": True
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "error": f"Requirements parsing error: {e}"
            }
    
    def _determine_overall_status(self, results: Dict[str, Any]) -> str:
        """Determine overall validation status"""
        
        failed_checks = [name for name, result in results.items() 
                        if result.get("status") == "fail"]
        
        if failed_checks:
            return "fail"
        
        warning_checks = [name for name, result in results.items() 
                         if result.get("status") == "warning"]
        
        if warning_checks:
            return "warning"
        
        return "pass"
    
    def display_detailed_results(self, results: Dict[str, Any]):
        """Display detailed validation results"""
        
        console.print("\n" + "="*60)
        console.print("🔍 NIS PROJECT VALIDATION REPORT", style="bold blue")
        console.print("="*60)
        
        # Overall status
        overall = results["overall_status"]
        if overall == "pass":
            console.print("✅ OVERALL STATUS: PASS", style="bold green")
        elif overall == "warning":
            console.print("⚠️  OVERALL STATUS: WARNING", style="bold yellow")
        else:
            console.print("❌ OVERALL STATUS: FAIL", style="bold red")
        
        # Detailed results
        for check_name, result in results["results"].items():
            console.print(f"\n📋 {check_name}:", style="bold")
            
            if result.get("status") == "pass":
                console.print("   ✅ PASS", style="green")
            elif result.get("status") == "warning":
                console.print("   ⚠️  WARNING", style="yellow")
            else:
                console.print("   ❌ FAIL", style="red")
            
            # Show specific details
            if "missing_dirs" in result and result["missing_dirs"]:
                console.print(f"   Missing directories: {result['missing_dirs']}")
            if "missing_files" in result and result["missing_files"]:
                console.print(f"   Missing files: {result['missing_files']}")
            if "error" in result:
                console.print(f"   Error: {result['error']}", style="red")

def validate_project(project_path: str = ".") -> bool:
    """Main validation function"""
    
    validator = NISValidator(Path(project_path))
    results = validator.validate_project()
    validator.display_detailed_results(results)
    
    return results["overall_status"] == "pass"

def main():
    """CLI entry point"""
    import sys
    
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    if validate_project(project_path):
        console.print("\n✅ Project validation completed successfully!", style="bold green")
        sys.exit(0)
    else:
        console.print("\n❌ Project validation failed!", style="bold red")
        sys.exit(1)

if __name__ == "__main__":
    main()
