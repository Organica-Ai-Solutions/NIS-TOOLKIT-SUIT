#!/usr/bin/env python3
"""
NIS Core Toolkit - Enhanced Project Validation
Comprehensive validation for NIS Protocol v3 compliance and integration
"""

import yaml
import json
import ast
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import asyncio
import sys

console = Console()

class NISProtocolV3Validator:
    """Enhanced NIS Protocol v3 compliance validator"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(".")
        self.validation_results = []
        self.errors = []
        self.warnings = []
        self.info = []
        
        # NIS Protocol v3 requirements
        self.required_interfaces = {
            "consciousness": ["NISConsciousnessInterface", "ConsciousnessConfig"],
            "kan": ["NISKANInterface", "KANConfig", "KANResult"],
            "base_agent": ["BaseNISAgent", "observe", "decide", "act"]
        }
        
        self.protocol_v3_features = {
            "mathematical_guarantees": ["spline_order", "interpretability_threshold", "convergence_guarantees"],
            "consciousness_processing": ["meta_cognitive_processing", "bias_detection", "self_reflection"],
            "ecosystem_integration": ["integration_connector", "message_handlers", "project_registry"]
        }
    
    async def validate_project_comprehensive(self) -> Dict[str, Any]:
        """Run comprehensive NIS Protocol v3 validation"""
        
        console.print("🧠 Running NIS Protocol v3 Comprehensive Validation...", style="bold blue")
        console.print("=" * 70)
        
        validation_suites = [
            ("Project Structure", self._validate_enhanced_structure),
            ("Configuration", self._validate_enhanced_config),
            ("Agent Compliance", self._validate_agent_compliance),
            ("NIS Protocol v3", self._validate_protocol_v3_compliance),
            ("KAN Integration", self._validate_kan_integration),
            ("Consciousness Interface", self._validate_consciousness_interface),
            ("Ecosystem Connectivity", self._validate_ecosystem_integration),
            ("Dependencies", self._validate_enhanced_dependencies),
            ("Code Quality", self._validate_code_quality),
            ("Performance Standards", self._validate_performance_standards)
        ]
        
        results = {}
        overall_score = 0
        max_score = len(validation_suites) * 100
        
        for suite_name, validation_func in track(validation_suites, description="Validating..."):
            try:
                result = await validation_func()
                results[suite_name] = result
                
                # Calculate scores
                if result.get("status") == "pass":
                    overall_score += 100
                    console.print(f"✅ {suite_name}: PASS ({result.get('score', 100)}/100)", style="green")
                elif result.get("status") == "warning":
                    overall_score += result.get('score', 70)
                    console.print(f"⚠️  {suite_name}: WARNING ({result.get('score', 70)}/100)", style="yellow")
                else:
                    console.print(f"❌ {suite_name}: FAIL ({result.get('score', 0)}/100)", style="red")
                    
            except Exception as e:
                results[suite_name] = {"status": "error", "error": str(e), "score": 0}
                console.print(f"💥 {suite_name}: ERROR - {e}", style="red")
        
        # Generate comprehensive report
        final_score = (overall_score / max_score) * 100
        results["validation_summary"] = {
            "overall_score": final_score,
            "protocol_compliance": self._calculate_protocol_compliance(results),
            "agi_readiness": self._calculate_agi_readiness(results),
            "ecosystem_compatibility": self._calculate_ecosystem_compatibility(results),
            "recommendations": self._generate_recommendations(results)
        }
        
        # Display final results
        self._display_validation_summary(results)
        
        return results
    
    async def _validate_enhanced_structure(self) -> Dict[str, Any]:
        """Enhanced project structure validation"""
        
        required_structure = {
            "agents/": "Directory for intelligent agents",
            "config/": "Configuration directory",
            "config/project.yaml": "Main project configuration",
            "config/agents.yaml": "Agent configuration",
            "config/integration.yaml": "Ecosystem integration config",
            "main.py": "Main entry point",
            "requirements.txt": "Python dependencies",
            "templates/": "Agent and system templates",
            "tests/": "Test suites",
            "logs/": "Logging directory"
        }
        
        nis_v3_structure = {
            "nis_v3/": "NIS Protocol v3 integration",
            "nis_v3/consciousness/": "Consciousness interface implementations",
            "nis_v3/kan/": "KAN network implementations",
            "nis_v3/coordination/": "Multi-agent coordination",
            "integrations/": "Ecosystem integration modules"
        }
        
        score = 100
        missing = []
        present = []
        
        # Check required structure
        for path, description in required_structure.items():
            full_path = self.project_root / path
            if full_path.exists():
                present.append(path)
            else:
                missing.append(f"{path} - {description}")
                score -= 8
        
        # Check NIS v3 specific structure
        v3_score = 0
        for path, description in nis_v3_structure.items():
            full_path = self.project_root / path
            if full_path.exists():
                v3_score += 20
                present.append(f"{path} (NIS v3)")
        
        # Bonus for NIS v3 compliance
        if v3_score >= 60:
            score += 10
        
        status = "pass" if score >= 80 else "warning" if score >= 60 else "fail"
        
        return {
            "status": status,
            "score": min(score, 100),
            "missing_required": missing,
            "present_structure": present,
            "nis_v3_compliance": v3_score,
            "details": f"Found {len(present)} required elements, missing {len(missing)}"
        }
    
    async def _validate_enhanced_config(self) -> Dict[str, Any]:
        """Enhanced configuration validation"""
        
        config_checks = []
        score = 100
        
        # Main project config
        project_config_path = self.project_root / "config/project.yaml"
        if project_config_path.exists():
            try:
                with open(project_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check NIS Protocol version
                if config.get("nis_protocol_version") == "3.0":
                    config_checks.append("✅ NIS Protocol v3.0 specified")
                else:
                    config_checks.append("❌ NIS Protocol v3.0 not specified")
                    score -= 20
                
                # Check consciousness configuration
                if "consciousness" in config:
                    consciousness_config = config["consciousness"]
                    required_consciousness = ["meta_cognitive_processing", "bias_detection", "introspection_depth"]
                    for req in required_consciousness:
                        if req in consciousness_config:
                            config_checks.append(f"✅ Consciousness config: {req}")
                        else:
                            config_checks.append(f"❌ Missing consciousness config: {req}")
                            score -= 5
                else:
                    config_checks.append("❌ No consciousness configuration found")
                    score -= 15
                
                # Check KAN configuration
                if "kan" in config:
                    kan_config = config["kan"]
                    required_kan = ["spline_order", "grid_size", "interpretability_threshold"]
                    for req in required_kan:
                        if req in kan_config:
                            config_checks.append(f"✅ KAN config: {req}")
                        else:
                            config_checks.append(f"❌ Missing KAN config: {req}")
                            score -= 5
                else:
                    config_checks.append("❌ No KAN configuration found")
                    score -= 15
                
            except Exception as e:
                config_checks.append(f"❌ Error parsing project.yaml: {e}")
                score -= 30
        else:
            config_checks.append("❌ project.yaml not found")
            score -= 40
        
        status = "pass" if score >= 80 else "warning" if score >= 60 else "fail"
        
        return {
            "status": status,
            "score": max(score, 0),
            "config_checks": config_checks,
            "details": f"Configuration validation completed with {len(config_checks)} checks"
        }
    
    async def _validate_agent_compliance(self) -> Dict[str, Any]:
        """Validate agent compliance with NIS standards"""
        
        agents_dir = self.project_root / "agents"
        if not agents_dir.exists():
            return {
                "status": "fail",
                "score": 0,
                "error": "No agents directory found",
                "details": "Agents directory is required for NIS projects"
            }
        
        agent_files = list(agents_dir.rglob("*.py"))
        compliant_agents = []
        non_compliant_agents = []
        score = 100
        
        for agent_file in agent_files:
            if agent_file.name.startswith("__"):
                continue
                
            try:
                # Parse agent file
                with open(agent_file, 'r') as f:
                    content = f.read()
                
                # Check for BaseNISAgent inheritance
                if "BaseNISAgent" in content:
                    # Check for required methods
                    required_methods = ["observe", "decide", "act"]
                    methods_found = []
                    
                    for method in required_methods:
                        if f"async def {method}(" in content:
                            methods_found.append(method)
                    
                    if len(methods_found) == 3:
                        compliant_agents.append({
                            "file": str(agent_file.relative_to(self.project_root)),
                            "methods": methods_found,
                            "nis_compliant": True
                        })
                    else:
                        non_compliant_agents.append({
                            "file": str(agent_file.relative_to(self.project_root)),
                            "methods": methods_found,
                            "missing": [m for m in required_methods if m not in methods_found]
                        })
                        score -= 10
                else:
                    non_compliant_agents.append({
                        "file": str(agent_file.relative_to(self.project_root)),
                        "issue": "Does not inherit from BaseNISAgent"
                    })
                    score -= 15
                    
            except Exception as e:
                non_compliant_agents.append({
                    "file": str(agent_file.relative_to(self.project_root)),
                    "error": str(e)
                })
                score -= 5
        
        total_agents = len(agent_files)
        compliance_rate = len(compliant_agents) / max(total_agents, 1)
        
        status = "pass" if compliance_rate >= 0.8 else "warning" if compliance_rate >= 0.5 else "fail"
        
        return {
            "status": status,
            "score": max(score, 0),
            "total_agents": total_agents,
            "compliant_agents": len(compliant_agents),
            "compliance_rate": compliance_rate,
            "compliant_details": compliant_agents,
            "non_compliant_details": non_compliant_agents,
            "details": f"Found {total_agents} agents, {len(compliant_agents)} NIS-compliant"
        }
    
    async def _validate_protocol_v3_compliance(self) -> Dict[str, Any]:
        """Validate NIS Protocol v3 specific compliance"""
        
        v3_features_found = []
        missing_features = []
        score = 100
        
        # Check for consciousness interface
        consciousness_files = list(self.project_root.rglob("*consciousness*.py"))
        if consciousness_files:
            v3_features_found.append("Consciousness interface files found")
            # Check for specific consciousness classes
            for cf in consciousness_files:
                try:
                    with open(cf, 'r') as f:
                        content = f.read()
                    if "NISConsciousnessInterface" in content:
                        v3_features_found.append("NISConsciousnessInterface implementation")
                    if "ConsciousnessConfig" in content:
                        v3_features_found.append("ConsciousnessConfig found")
                except Exception:
                    pass
        else:
            missing_features.append("No consciousness interface found")
            score -= 25
        
        # Check for KAN interface
        kan_files = list(self.project_root.rglob("*kan*.py"))
        if kan_files:
            v3_features_found.append("KAN interface files found")
            for kf in kan_files:
                try:
                    with open(kf, 'r') as f:
                        content = f.read()
                    if "NISKANInterface" in content:
                        v3_features_found.append("NISKANInterface implementation")
                    if "spline" in content.lower():
                        v3_features_found.append("Spline-based processing detected")
                    if "interpretability" in content.lower():
                        v3_features_found.append("Interpretability features found")
                except Exception:
                    pass
        else:
            missing_features.append("No KAN interface found")
            score -= 25
        
        # Check for integration connector
        integration_files = list(self.project_root.rglob("*integration*.py"))
        if integration_files:
            v3_features_found.append("Integration files found")
        else:
            missing_features.append("No ecosystem integration found")
            score -= 15
        
        # Check for mathematical guarantees
        math_keywords = ["mathematical_proof", "convergence", "spline_coefficient", "interpretability_score"]
        math_features = 0
        for keyword in math_keywords:
            files_with_keyword = []
            for py_file in self.project_root.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        if keyword in f.read():
                            files_with_keyword.append(py_file)
                            break
                except Exception:
                    continue
            if files_with_keyword:
                math_features += 1
                v3_features_found.append(f"Mathematical feature: {keyword}")
        
        if math_features < 2:
            missing_features.append("Insufficient mathematical guarantees implementation")
            score -= 10
        
        protocol_compliance = len(v3_features_found) / (len(v3_features_found) + len(missing_features))
        status = "pass" if protocol_compliance >= 0.75 else "warning" if protocol_compliance >= 0.5 else "fail"
        
        return {
            "status": status,
            "score": max(score, 0),
            "protocol_compliance": protocol_compliance,
            "v3_features_found": v3_features_found,
            "missing_features": missing_features,
            "details": f"Protocol v3 compliance: {protocol_compliance:.1%}"
        }
    
    async def _validate_kan_integration(self) -> Dict[str, Any]:
        """Validate KAN network integration"""
        
        kan_score = 100
        kan_features = []
        kan_issues = []
        
        # Look for KAN-related files and implementations
        kan_patterns = ["kan", "spline", "kolmogorov", "arnold"]
        kan_files = []
        
        for pattern in kan_patterns:
            pattern_files = list(self.project_root.rglob(f"*{pattern}*.py"))
            kan_files.extend(pattern_files)
        
        kan_files = list(set(kan_files))  # Remove duplicates
        
        if not kan_files:
            return {
                "status": "fail",
                "score": 0,
                "error": "No KAN implementation found",
                "details": "KAN networks are required for NIS Protocol v3"
            }
        
        # Analyze KAN implementations
        for kan_file in kan_files:
            try:
                with open(kan_file, 'r') as f:
                    content = f.read()
                
                # Check for key KAN features
                if "spline_order" in content:
                    kan_features.append(f"Spline configuration in {kan_file.name}")
                if "interpretability" in content.lower():
                    kan_features.append(f"Interpretability support in {kan_file.name}")
                if "mathematical_proof" in content:
                    kan_features.append(f"Mathematical proofs in {kan_file.name}")
                if "convergence" in content:
                    kan_features.append(f"Convergence guarantees in {kan_file.name}")
                
                # Check for potential issues
                if "TODO" in content:
                    kan_issues.append(f"TODO items in {kan_file.name}")
                    kan_score -= 5
                if "placeholder" in content.lower():
                    kan_issues.append(f"Placeholder code in {kan_file.name}")
                    kan_score -= 10
                    
            except Exception as e:
                kan_issues.append(f"Error reading {kan_file.name}: {e}")
                kan_score -= 5
        
        # Score based on feature completeness
        feature_score = len(kan_features) * 5
        kan_score = min(kan_score + feature_score, 100)
        
        status = "pass" if kan_score >= 80 else "warning" if kan_score >= 60 else "fail"
        
        return {
            "status": status,
            "score": kan_score,
            "kan_files_found": len(kan_files),
            "kan_features": kan_features,
            "kan_issues": kan_issues,
            "details": f"Found {len(kan_files)} KAN-related files with {len(kan_features)} features"
        }
    
    async def _validate_consciousness_interface(self) -> Dict[str, Any]:
        """Validate consciousness interface implementation"""
        
        consciousness_score = 100
        consciousness_features = []
        consciousness_issues = []
        
        # Look for consciousness-related files
        consciousness_files = list(self.project_root.rglob("*consciousness*.py"))
        
        if not consciousness_files:
            return {
                "status": "warning",
                "score": 50,
                "warning": "No consciousness interface found",
                "details": "Consciousness interface is recommended for advanced NIS v3 features"
            }
        
        for cons_file in consciousness_files:
            try:
                with open(cons_file, 'r') as f:
                    content = f.read()
                
                # Check for consciousness features
                required_features = [
                    ("meta_cognitive_processing", "Meta-cognitive processing"),
                    ("bias_detection", "Bias detection"),
                    ("self_reflection", "Self-reflection"),
                    ("introspection", "Introspection capabilities"),
                    ("consciousness_state", "Consciousness state tracking")
                ]
                
                for feature, description in required_features:
                    if feature in content:
                        consciousness_features.append(f"{description} in {cons_file.name}")
                    else:
                        consciousness_issues.append(f"Missing {description} in {cons_file.name}")
                        consciousness_score -= 10
                
                # Check for advanced features
                advanced_features = ["emotional_awareness", "attention_tracking", "cognitive_load"]
                for feature in advanced_features:
                    if feature in content:
                        consciousness_features.append(f"Advanced: {feature} in {cons_file.name}")
                        consciousness_score += 5
                        
            except Exception as e:
                consciousness_issues.append(f"Error reading {cons_file.name}: {e}")
                consciousness_score -= 10
        
        status = "pass" if consciousness_score >= 80 else "warning" if consciousness_score >= 60 else "fail"
        
        return {
            "status": status,
            "score": min(consciousness_score, 100),
            "consciousness_files": len(consciousness_files),
            "consciousness_features": consciousness_features,
            "consciousness_issues": consciousness_issues,
            "details": f"Consciousness validation: {len(consciousness_features)} features found"
        }
    
    async def _validate_ecosystem_integration(self) -> Dict[str, Any]:
        """Validate ecosystem integration capabilities"""
        
        integration_score = 100
        integration_features = []
        integration_issues = []
        
        # Check for integration connector
        integration_files = list(self.project_root.rglob("*integration*.py"))
        connector_files = list(self.project_root.rglob("*connector*.py"))
        
        all_integration_files = integration_files + connector_files
        
        if not all_integration_files:
            return {
                "status": "warning",
                "score": 30,
                "warning": "No ecosystem integration found",
                "details": "Ecosystem integration enables connection to NIS-X, NIS-DRONE, etc."
            }
        
        for int_file in all_integration_files:
            try:
                with open(int_file, 'r') as f:
                    content = f.read()
                
                # Check for integration features
                if "NISIntegrationConnector" in content:
                    integration_features.append(f"NIS Integration Connector in {int_file.name}")
                if "websocket" in content.lower():
                    integration_features.append(f"WebSocket support in {int_file.name}")
                if "authentication" in content.lower():
                    integration_features.append(f"Authentication support in {int_file.name}")
                if "heartbeat" in content.lower():
                    integration_features.append(f"Heartbeat monitoring in {int_file.name}")
                
                # Check for ecosystem projects
                ecosystem_projects = ["nis-x", "nis-drone", "archaeological", "sparknova", "orion"]
                for project in ecosystem_projects:
                    if project in content.lower():
                        integration_features.append(f"Support for {project} in {int_file.name}")
                        
            except Exception as e:
                integration_issues.append(f"Error reading {int_file.name}: {e}")
                integration_score -= 10
        
        # Bonus for comprehensive integration
        if len(integration_features) >= 5:
            integration_score += 10
        
        status = "pass" if integration_score >= 70 else "warning" if integration_score >= 50 else "fail"
        
        return {
            "status": status,
            "score": min(integration_score, 100),
            "integration_files": len(all_integration_files),
            "integration_features": integration_features,
            "integration_issues": integration_issues,
            "details": f"Integration validation: {len(integration_features)} features found"
        }
    
    async def _validate_enhanced_dependencies(self) -> Dict[str, Any]:
        """Enhanced dependency validation"""
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            return {
                "status": "fail",
                "score": 0,
                "error": "requirements.txt not found",
                "details": "Requirements file is necessary for dependency management"
            }
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = f.read()
            
            # Check for essential NIS dependencies
            essential_deps = {
                "nis-core-toolkit": "NIS Core Toolkit",
                "nis-agent-toolkit": "NIS Agent Toolkit", 
                "pydantic": "Data validation",
                "pyyaml": "Configuration handling",
                "rich": "Console output",
                "asyncio": "Async processing"
            }
            
            deps_found = []
            deps_missing = []
            
            for dep, description in essential_deps.items():
                if dep in requirements.lower():
                    deps_found.append(f"{dep} - {description}")
                else:
                    deps_missing.append(f"{dep} - {description}")
            
            # Check for advanced dependencies
            advanced_deps = ["torch", "transformers", "numpy", "scipy", "redis", "fastapi"]
            advanced_found = []
            
            for dep in advanced_deps:
                if dep in requirements.lower():
                    advanced_found.append(dep)
            
            score = (len(deps_found) / len(essential_deps)) * 80
            if advanced_found:
                score += min(len(advanced_found) * 3, 20)
            
            status = "pass" if score >= 80 else "warning" if score >= 60 else "fail"
            
            return {
                "status": status,
                "score": min(int(score), 100),
                "essential_deps_found": deps_found,
                "essential_deps_missing": deps_missing,
                "advanced_deps_found": advanced_found,
                "details": f"Dependencies: {len(deps_found)}/{len(essential_deps)} essential, {len(advanced_found)} advanced"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "score": 0,
                "error": f"Error reading requirements.txt: {e}",
                "details": "Could not validate dependencies"
            }
    
    async def _validate_code_quality(self) -> Dict[str, Any]:
        """Validate code quality and best practices"""
        
        python_files = list(self.project_root.rglob("*.py"))
        quality_score = 100
        quality_issues = []
        quality_features = []
        
        for py_file in python_files:
            if py_file.name.startswith("__"):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Check for good practices
                if '"""' in content or "'''" in content:
                    quality_features.append(f"Docstrings in {py_file.name}")
                
                if "async def" in content:
                    quality_features.append(f"Async support in {py_file.name}")
                
                if "typing" in content and "import" in content:
                    quality_features.append(f"Type hints in {py_file.name}")
                
                # Check for issues
                if "TODO" in content:
                    todo_count = content.count("TODO")
                    if todo_count > 3:
                        quality_issues.append(f"Many TODOs ({todo_count}) in {py_file.name}")
                        quality_score -= min(todo_count, 10)
                
                if "print(" in content and py_file.name != "main.py":
                    quality_issues.append(f"Print statements in {py_file.name} (use logging)")
                    quality_score -= 2
                
                # Try basic syntax validation
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    quality_issues.append(f"Syntax error in {py_file.name}: {e}")
                    quality_score -= 20
                    
            except Exception as e:
                quality_issues.append(f"Error reading {py_file.name}: {e}")
                quality_score -= 5
        
        status = "pass" if quality_score >= 80 else "warning" if quality_score >= 60 else "fail"
        
        return {
            "status": status,
            "score": max(quality_score, 0),
            "python_files_checked": len(python_files),
            "quality_features": quality_features,
            "quality_issues": quality_issues,
            "details": f"Code quality: {len(quality_features)} good practices, {len(quality_issues)} issues"
        }
    
    async def _validate_performance_standards(self) -> Dict[str, Any]:
        """Validate performance and efficiency standards"""
        
        performance_score = 100
        performance_features = []
        performance_issues = []
        
        # Check for performance-related implementations
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Good performance practices
                if "asyncio" in content:
                    performance_features.append(f"Async processing in {py_file.name}")
                
                if "cache" in content.lower():
                    performance_features.append(f"Caching implementation in {py_file.name}")
                
                if "pool" in content.lower():
                    performance_features.append(f"Connection pooling in {py_file.name}")
                
                # Performance concerns
                if "time.sleep(" in content:
                    performance_issues.append(f"Blocking sleep in {py_file.name}")
                    performance_score -= 10
                
                if content.count("for ") > 5:  # Many loops might indicate inefficiency
                    performance_issues.append(f"Many loops in {py_file.name} - review efficiency")
                    performance_score -= 5
                    
            except Exception:
                continue
        
        # Check for test files (performance testing)
        test_files = list(self.project_root.rglob("test_*.py"))
        if test_files:
            performance_features.append(f"Test suite present ({len(test_files)} files)")
        else:
            performance_issues.append("No test files found")
            performance_score -= 15
        
        status = "pass" if performance_score >= 80 else "warning" if performance_score >= 60 else "fail"
        
        return {
            "status": status,
            "score": max(performance_score, 0),
            "performance_features": performance_features,
            "performance_issues": performance_issues,
            "details": f"Performance: {len(performance_features)} optimizations, {len(performance_issues)} concerns"
        }
    
    def _calculate_protocol_compliance(self, results: Dict[str, Any]) -> float:
        """Calculate overall NIS Protocol compliance score"""
        
        key_areas = ["NIS Protocol v3", "KAN Integration", "Consciousness Interface", "Agent Compliance"]
        total_score = 0
        total_weight = 0
        
        for area in key_areas:
            if area in results and "score" in results[area]:
                total_score += results[area]["score"]
                total_weight += 100
        
        return (total_score / total_weight) if total_weight > 0 else 0.0
    
    def _calculate_agi_readiness(self, results: Dict[str, Any]) -> float:
        """Calculate AGI readiness score"""
        
        agi_factors = {
            "KAN Integration": 0.3,
            "Consciousness Interface": 0.3,
            "Agent Compliance": 0.2,
            "Ecosystem Connectivity": 0.2
        }
        
        agi_score = 0
        for factor, weight in agi_factors.items():
            if factor in results and "score" in results[factor]:
                agi_score += (results[factor]["score"] / 100) * weight
        
        return agi_score
    
    def _calculate_ecosystem_compatibility(self, results: Dict[str, Any]) -> float:
        """Calculate ecosystem compatibility score"""
        
        compatibility_factors = ["Ecosystem Connectivity", "Dependencies", "Code Quality"]
        total_score = 0
        
        for factor in compatibility_factors:
            if factor in results and "score" in results[factor]:
                total_score += results[factor]["score"]
        
        return total_score / (len(compatibility_factors) * 100)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Protocol compliance recommendations
        if results.get("NIS Protocol v3", {}).get("score", 0) < 80:
            recommendations.append("🧠 Implement missing NIS Protocol v3 features (consciousness, KAN)")
        
        # KAN integration recommendations
        if results.get("KAN Integration", {}).get("score", 0) < 70:
            recommendations.append("🧮 Add KAN network implementation for mathematical reasoning")
        
        # Consciousness recommendations
        if results.get("Consciousness Interface", {}).get("score", 0) < 70:
            recommendations.append("🎯 Implement consciousness interface for meta-cognitive processing")
        
        # Agent compliance recommendations
        if results.get("Agent Compliance", {}).get("score", 0) < 80:
            recommendations.append("🤖 Ensure all agents inherit from BaseNISAgent with observe/decide/act")
        
        # Code quality recommendations
        if results.get("Code Quality", {}).get("score", 0) < 70:
            recommendations.append("📝 Improve code quality: add docstrings, type hints, reduce TODOs")
        
        # Performance recommendations
        if results.get("Performance Standards", {}).get("score", 0) < 70:
            recommendations.append("⚡ Optimize performance: use async processing, add caching")
        
        # Ecosystem recommendations
        if results.get("Ecosystem Connectivity", {}).get("score", 0) < 60:
            recommendations.append("🔗 Add ecosystem integration for NIS-X, NIS-DRONE connectivity")
        
        return recommendations
    
    def _display_validation_summary(self, results: Dict[str, Any]):
        """Display comprehensive validation summary"""
        
        summary = results.get("validation_summary", {})
        
        # Create summary panel
        summary_content = f"""
[bold]NIS Protocol v3 Validation Results[/bold]

[green]Overall Score:[/green] {summary.get('overall_score', 0):.1f}/100
[blue]Protocol Compliance:[/blue] {summary.get('protocol_compliance', 0):.1%}
[purple]AGI Readiness:[/purple] {summary.get('agi_readiness', 0):.1%}
[cyan]Ecosystem Compatibility:[/cyan] {summary.get('ecosystem_compatibility', 0):.1%}

[bold yellow]Recommendations:[/bold yellow]
"""
        
        for i, rec in enumerate(summary.get('recommendations', []), 1):
            summary_content += f"\n{i}. {rec}"
        
        console.print(Panel(summary_content, title="🧠 NIS Validation Summary", border_style="blue"))


def validate_project(fix_issues: bool = False, strict_mode: bool = False) -> Dict[str, Any]:
    """Main validation function for CLI"""
    
    validator = NISProtocolV3Validator()
    
    # Run async validation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(validator.validate_project_comprehensive())
    finally:
        loop.close()
    
    return results
