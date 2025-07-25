#!/usr/bin/env python3
"""
NIS-CORE-TOOLKIT: Intelligent Multi-Agent System CLI

The ultimate CLI for integrating NIS Protocol into any project.
Analyzes your codebase, understands your domain, and creates
intelligent agent systems tailored to your specific needs.

Commands:
    analyze     - Analyze existing project for NIS integration opportunities
    integrate   - Intelligently integrate NIS Protocol into any project
    init        - Initialize new NIS-powered project from scratch
    agents      - Manage and coordinate intelligent agents
    orchestrate - Set up real-time multi-agent coordination
    deploy      - Deploy NIS system to any infrastructure
    monitor     - Real-time consciousness and performance monitoring
    optimize    - Intelligent system optimization and tuning
"""

import asyncio
import click
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import subprocess
import shutil

# Add the toolkit to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.integration_connector import NISIntegrationConnector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('nis-core-cli')

class Colors:
    """ANSI color codes for beautiful CLI output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_banner():
    """Print the amazing NIS CLI banner"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🧠 NIS-CORE-TOOLKIT: Intelligent Multi-Agent System CLI    ║
║                                                               ║
║   Transform ANY project into an intelligent system with      ║
║   consciousness-aware agents and mathematical guarantees     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.END}
"""
    print(banner)

class ProjectAnalyzer:
    """Intelligent project analysis for NIS integration"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.analysis_results = {}
        
    async def analyze_project(self) -> Dict[str, Any]:
        """Perform comprehensive project analysis"""
        print(f"{Colors.BLUE}🔍 Analyzing project: {self.project_path}{Colors.END}")
        
        analysis = {
            "project_type": await self._detect_project_type(),
            "language_stack": await self._analyze_languages(),
            "frameworks": await self._detect_frameworks(),
            "domain_hints": await self._infer_domain(),
            "complexity": await self._assess_complexity(),
            "nis_opportunities": await self._identify_nis_opportunities(),
            "recommended_agents": await self._recommend_agents(),
            "integration_strategy": await self._plan_integration(),
            "estimated_effort": await self._estimate_effort()
        }
        
        self.analysis_results = analysis
        return analysis
    
    async def _detect_project_type(self) -> str:
        """Detect the type of project"""
        indicators = {
            "web_app": ["package.json", "requirements.txt", "index.html", "app.py", "server.js"],
            "api_service": ["fastapi", "flask", "express", "spring", "api", "swagger"],
            "data_science": ["jupyter", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
            "mobile_app": ["android", "ios", "react-native", "flutter", "ionic"],
            "desktop_app": ["electron", "tkinter", "qt", "javafx", "wpf"],
            "ml_pipeline": ["mlflow", "kubeflow", "airflow", "pipeline", "model"],
            "research": ["research", "experiment", "analysis", "study", "paper"],
            "enterprise": ["microservices", "kubernetes", "docker", "enterprise", "production"]
        }
        
        file_contents = []
        for file_path in self.project_path.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size < 1024 * 1024:  # < 1MB
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
                    file_contents.append(content)
                except:
                    pass
        
        all_content = " ".join(file_contents)
        
        scores = {}
        for project_type, keywords in indicators.items():
            score = sum(1 for keyword in keywords if keyword in all_content)
            scores[project_type] = score
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else "general"
    
    async def _analyze_languages(self) -> List[str]:
        """Detect programming languages used"""
        language_extensions = {
            ".py": "python",
            ".js": "javascript", 
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby",
            ".scala": "scala",
            ".kt": "kotlin",
            ".swift": "swift",
            ".r": "r",
            ".jl": "julia"
        }
        
        languages = set()
        for file_path in self.project_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in language_extensions:
                    languages.add(language_extensions[ext])
        
        return list(languages)
    
    async def _detect_frameworks(self) -> List[str]:
        """Detect frameworks and libraries used"""
        framework_indicators = {
            "react": ["react", "jsx", "package.json"],
            "vue": ["vue", "package.json"],
            "angular": ["angular", "@angular"],
            "django": ["django", "settings.py", "manage.py"],
            "flask": ["flask", "app.py"],
            "fastapi": ["fastapi", "main.py"],
            "express": ["express", "package.json"],
            "spring": ["spring", "pom.xml", "gradle"],
            "tensorflow": ["tensorflow", "keras"],
            "pytorch": ["torch", "pytorch"],
            "scikit-learn": ["sklearn", "scikit-learn"],
            "pandas": ["pandas", "dataframe"],
            "numpy": ["numpy", "np."],
            "docker": ["Dockerfile", "docker-compose"],
            "kubernetes": ["deployment.yaml", "service.yaml"]
        }
        
        detected_frameworks = []
        
        # Check files for framework indicators
        for file_path in self.project_path.rglob("*"):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
                    filename = file_path.name.lower()
                    
                    for framework, indicators in framework_indicators.items():
                        for indicator in indicators:
                            if indicator in content or indicator in filename:
                                if framework not in detected_frameworks:
                                    detected_frameworks.append(framework)
                                break
                except:
                    pass
        
        return detected_frameworks
    
    async def _infer_domain(self) -> Dict[str, float]:
        """Infer the application domain with confidence scores"""
        domain_keywords = {
            "healthcare": ["patient", "medical", "diagnosis", "treatment", "clinical", "hospital", "doctor", "health"],
            "finance": ["trading", "investment", "portfolio", "risk", "market", "financial", "bank", "payment"],
            "ecommerce": ["product", "cart", "order", "payment", "shipping", "customer", "inventory", "catalog"],
            "education": ["student", "course", "lesson", "grade", "exam", "learning", "teacher", "education"],
            "gaming": ["game", "player", "score", "level", "character", "quest", "match", "gaming"],
            "iot": ["sensor", "device", "iot", "telemetry", "monitoring", "automation", "embedded"],
            "research": ["research", "experiment", "analysis", "study", "data", "hypothesis", "publication"],
            "media": ["content", "video", "image", "media", "streaming", "broadcast", "social"],
            "logistics": ["shipping", "delivery", "warehouse", "supply", "logistics", "transport", "freight"],
            "manufacturing": ["production", "manufacturing", "factory", "assembly", "quality", "industrial"]
        }
        
        file_contents = []
        for file_path in self.project_path.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size < 1024 * 1024:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
                    file_contents.append(content)
                except:
                    pass
        
        all_content = " ".join(file_contents)
        
        domain_scores = {}
        total_words = len(all_content.split())
        
        for domain, keywords in domain_keywords.items():
            matches = sum(all_content.count(keyword) for keyword in keywords)
            confidence = min(matches / max(total_words * 0.001, 1), 1.0)  # Normalize
            if confidence > 0.1:  # Only include domains with reasonable confidence
                domain_scores[domain] = confidence
        
        return domain_scores
    
    async def _assess_complexity(self) -> Dict[str, Any]:
        """Assess project complexity"""
        file_count = 0
        total_lines = 0
        avg_file_size = 0
        
        for file_path in self.project_path.rglob("*"):
            if file_path.is_file():
                file_count += 1
                try:
                    lines = len(file_path.read_text(encoding='utf-8', errors='ignore').splitlines())
                    total_lines += lines
                except:
                    pass
        
        if file_count > 0:
            avg_file_size = total_lines / file_count
        
        complexity_level = "low"
        if total_lines > 10000 or file_count > 100:
            complexity_level = "high"
        elif total_lines > 2000 or file_count > 20:
            complexity_level = "medium"
        
        return {
            "level": complexity_level,
            "file_count": file_count,
            "total_lines": total_lines,
            "avg_file_size": int(avg_file_size)
        }
    
    async def _identify_nis_opportunities(self) -> List[Dict[str, Any]]:
        """Identify NIS integration opportunities"""
        opportunities = []
        
        # Data processing opportunities
        if any(fw in self.analysis_results.get("frameworks", []) for fw in ["pandas", "numpy", "tensorflow", "pytorch"]):
            opportunities.append({
                "type": "data_intelligence",
                "description": "Add intelligent data analysis agents with consciousness-aware processing",
                "impact": "high",
                "effort": "medium"
            })
        
        # API enhancement opportunities
        if any(fw in self.analysis_results.get("frameworks", []) for fw in ["fastapi", "flask", "express", "spring"]):
            opportunities.append({
                "type": "api_intelligence",
                "description": "Transform API endpoints into intelligent multi-agent coordination points",
                "impact": "high", 
                "effort": "low"
            })
        
        # Frontend intelligence
        if any(fw in self.analysis_results.get("frameworks", []) for fw in ["react", "vue", "angular"]):
            opportunities.append({
                "type": "frontend_intelligence",
                "description": "Add intelligent user experience with adaptive agents",
                "impact": "medium",
                "effort": "medium"
            })
        
        # Domain-specific opportunities
        domain_scores = self.analysis_results.get("domain_hints", {})
        if "healthcare" in domain_scores:
            opportunities.append({
                "type": "healthcare_intelligence",
                "description": "Add medical AI agents with consciousness-aware safety protocols",
                "impact": "very_high",
                "effort": "high"
            })
        
        if "finance" in domain_scores:
            opportunities.append({
                "type": "financial_intelligence", 
                "description": "Add risk-aware financial agents with mathematical guarantees",
                "impact": "very_high",
                "effort": "high"
            })
        
        return opportunities
    
    async def _recommend_agents(self) -> List[Dict[str, Any]]:
        """Recommend specific agents for this project"""
        recommendations = []
        
        project_type = self.analysis_results.get("project_type", "general")
        frameworks = self.analysis_results.get("frameworks", [])
        domains = self.analysis_results.get("domain_hints", {})
        
        # Universal agents everyone needs
        recommendations.extend([
            {
                "type": "reasoning",
                "name": "intelligent_analyzer",
                "purpose": "Analyze and reason about project-specific data",
                "priority": "high"
            },
            {
                "type": "memory",
                "name": "knowledge_keeper",
                "purpose": "Store and retrieve project knowledge intelligently",
                "priority": "high"
            }
        ])
        
        # Data processing projects
        if any(fw in frameworks for fw in ["pandas", "numpy", "tensorflow", "pytorch"]):
            recommendations.append({
                "type": "vision",
                "name": "data_pattern_detector",
                "purpose": "Detect patterns and anomalies in data with consciousness",
                "priority": "high"
            })
        
        # Web applications
        if project_type == "web_app":
            recommendations.append({
                "type": "action",
                "name": "user_experience_optimizer",
                "purpose": "Optimize user interactions with intelligent responses",
                "priority": "medium"
            })
        
        # API services
        if project_type == "api_service":
            recommendations.append({
                "type": "action",
                "name": "api_orchestrator",
                "purpose": "Intelligently coordinate API responses and workflows",
                "priority": "high"
            })
        
        # Domain-specific agents
        if "healthcare" in domains:
            recommendations.append({
                "type": "reasoning",
                "name": "medical_intelligence",
                "purpose": "Provide medical reasoning with safety consciousness",
                "priority": "very_high"
            })
        
        if "finance" in domains:
            recommendations.append({
                "type": "reasoning", 
                "name": "risk_assessor",
                "purpose": "Assess financial risks with mathematical guarantees",
                "priority": "very_high"
            })
        
        return recommendations
    
    async def _plan_integration(self) -> Dict[str, Any]:
        """Plan the NIS integration strategy"""
        languages = self.analysis_results.get("language_stack", [])
        frameworks = self.analysis_results.get("frameworks", [])
        complexity = self.analysis_results.get("complexity", {})
        
        strategy = {
            "approach": "gradual",  # gradual, comprehensive, minimal
            "entry_points": [],
            "integration_files": [],
            "configuration": {},
            "deployment": "local"
        }
        
        # Determine approach based on complexity
        if complexity.get("level") == "high":
            strategy["approach"] = "gradual"
        elif complexity.get("level") == "low":
            strategy["approach"] = "comprehensive"
        
        # Find integration entry points
        if "python" in languages:
            if "fastapi" in frameworks:
                strategy["entry_points"].append("FastAPI middleware integration")
            elif "flask" in frameworks:
                strategy["entry_points"].append("Flask blueprint integration")
            elif "django" in frameworks:
                strategy["entry_points"].append("Django app integration")
            else:
                strategy["entry_points"].append("Python module integration")
        
        if "javascript" in languages or "typescript" in languages:
            if "express" in frameworks:
                strategy["entry_points"].append("Express middleware integration")
            elif "react" in frameworks:
                strategy["entry_points"].append("React component integration")
            else:
                strategy["entry_points"].append("Node.js module integration")
        
        return strategy
    
    async def _estimate_effort(self) -> Dict[str, Any]:
        """Estimate integration effort"""
        complexity = self.analysis_results.get("complexity", {})
        opportunities = self.analysis_results.get("nis_opportunities", [])
        
        base_hours = 4  # Minimum integration time
        
        # Adjust based on complexity
        if complexity.get("level") == "high":
            base_hours *= 3
        elif complexity.get("level") == "medium":
            base_hours *= 2
        
        # Adjust based on opportunities
        base_hours += len(opportunities) * 2
        
        return {
            "estimated_hours": base_hours,
            "phases": [
                {"name": "Analysis & Planning", "hours": 1},
                {"name": "Core Integration", "hours": base_hours // 2},
                {"name": "Agent Implementation", "hours": base_hours // 3},
                {"name": "Testing & Validation", "hours": base_hours // 6}
            ],
            "complexity_factor": complexity.get("level", "low")
        }

class NISIntegrator:
    """Intelligent NIS Protocol integrator"""
    
    def __init__(self, project_path: str, analysis: Dict[str, Any]):
        self.project_path = Path(project_path)
        self.analysis = analysis
        self.integration_plan = {}
        
    async def integrate_nis_protocol(self, integration_type: str = "smart") -> Dict[str, Any]:
        """Integrate NIS Protocol into the project"""
        print(f"{Colors.GREEN}🚀 Integrating NIS Protocol into project...{Colors.END}")
        
        results = {
            "success": True,
            "files_created": [],
            "files_modified": [],
            "agents_created": [],
            "configuration": {},
            "next_steps": []
        }
        
        try:
            # Create NIS directory structure
            await self._create_nis_structure()
            
            # Generate configuration
            await self._generate_configuration()
            
            # Create recommended agents
            await self._create_agents()
            
            # Add integration points
            await self._add_integration_points()
            
            # Create monitoring setup
            await self._setup_monitoring()
            
            # Generate documentation
            await self._generate_documentation()
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            logger.error(f"Integration failed: {e}")
        
        return results
    
    async def _create_nis_structure(self):
        """Create NIS directory structure"""
        structure = {
            "nis_system": {
                "agents": ["vision", "reasoning", "memory", "action"],
                "config": ["system.yaml", "agents.yaml", "monitoring.yaml"],
                "coordination": ["orchestrator.py", "workflows.py"],
                "monitoring": ["metrics.py", "dashboards.py"],
                "tests": ["test_agents.py", "test_integration.py"]
            }
        }
        
        base_path = self.project_path / "nis_system"
        base_path.mkdir(exist_ok=True)
        
        for category, items in structure["nis_system"].items():
            category_path = base_path / category
            category_path.mkdir(exist_ok=True)
            
            if category == "agents":
                for agent_type in items:
                    (category_path / f"{agent_type}_agent.py").touch()
            else:
                for item in items:
                    (category_path / item).touch()

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🧠 NIS-CORE-TOOLKIT: Intelligent Multi-Agent System CLI"""
    print_banner()

@cli.command()
@click.argument('project_path', default='.')
@click.option('--output', '-o', default='nis_analysis.json', help='Output file for analysis results')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed analysis')
async def analyze(project_path: str, output: str, detailed: bool):
    """🔍 Analyze existing project for NIS integration opportunities"""
    
    analyzer = ProjectAnalyzer(project_path)
    results = await analyzer.analyze_project()
    
    # Save results
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display summary
    print(f"\n{Colors.CYAN}{Colors.BOLD}📊 PROJECT ANALYSIS COMPLETE{Colors.END}")
    print(f"{Colors.BLUE}Project Type:{Colors.END} {results['project_type']}")
    print(f"{Colors.BLUE}Languages:{Colors.END} {', '.join(results['language_stack'])}")
    print(f"{Colors.BLUE}Frameworks:{Colors.END} {', '.join(results['frameworks'])}")
    print(f"{Colors.BLUE}Complexity:{Colors.END} {results['complexity']['level']}")
    
    if results['domain_hints']:
        print(f"\n{Colors.YELLOW}🎯 Detected Domains:{Colors.END}")
        for domain, confidence in results['domain_hints'].items():
            print(f"  • {domain}: {confidence:.1%} confidence")
    
    print(f"\n{Colors.GREEN}💡 NIS Integration Opportunities:{Colors.END}")
    for opp in results['nis_opportunities']:
        print(f"  • {opp['type']}: {opp['description']} (Impact: {opp['impact']})")
    
    print(f"\n{Colors.CYAN}🤖 Recommended Agents:{Colors.END}")
    for agent in results['recommended_agents']:
        print(f"  • {agent['name']} ({agent['type']}): {agent['purpose']}")
    
    print(f"\n{Colors.BOLD}⏱️  Estimated Integration Time: {results['estimated_effort']['estimated_hours']} hours{Colors.END}")
    print(f"{Colors.GREEN}📁 Analysis saved to: {output}{Colors.END}")

@cli.command()
@click.argument('project_path', default='.')
@click.option('--type', '-t', default='smart', help='Integration type: smart, minimal, comprehensive')
@click.option('--analysis', '-a', help='Use existing analysis file')
@click.option('--auto-deploy', is_flag=True, help='Automatically deploy after integration')
async def integrate(project_path: str, type: str, analysis: str, auto_deploy: bool):
    """🚀 Intelligently integrate NIS Protocol into any project"""
    
    # Load or perform analysis
    if analysis and Path(analysis).exists():
        with open(analysis, 'r') as f:
            analysis_results = json.load(f)
        print(f"{Colors.BLUE}📁 Using existing analysis: {analysis}{Colors.END}")
    else:
        print(f"{Colors.BLUE}🔍 Performing fresh project analysis...{Colors.END}")
        analyzer = ProjectAnalyzer(project_path)
        analysis_results = await analyzer.analyze_project()
    
    # Perform integration
    integrator = NISIntegrator(project_path, analysis_results)
    results = await integrator.integrate_nis_protocol(type)
    
    if results['success']:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ NIS Protocol Integration Successful!{Colors.END}")
        print(f"{Colors.GREEN}Files Created: {len(results['files_created'])}{Colors.END}")
        print(f"{Colors.GREEN}Agents Generated: {len(results['agents_created'])}{Colors.END}")
        
        if auto_deploy:
            print(f"\n{Colors.CYAN}🚀 Auto-deploying NIS system...{Colors.END}")
            # Auto-deploy logic here
            
    else:
        print(f"\n{Colors.RED}❌ Integration Failed: {results.get('error', 'Unknown error')}{Colors.END}")

@cli.command()
@click.argument('project_name')
@click.option('--domain', '-d', help='Project domain (healthcare, finance, research, etc.)')
@click.option('--template', '-t', default='universal', help='Project template to use')
@click.option('--language', '-l', default='python', help='Primary programming language')
async def init(project_name: str, domain: str, template: str, language: str):
    """🎯 Initialize new NIS-powered project from scratch"""
    
    print(f"{Colors.CYAN}🎯 Creating NIS-powered project: {project_name}{Colors.END}")
    
    project_path = Path(project_name)
    if project_path.exists():
        click.echo(f"{Colors.RED}❌ Project directory already exists!{Colors.END}")
        return
    
    # Create project structure
    project_path.mkdir()
    
    # Generate based on domain and language
    templates = {
        "python": {
            "files": ["main.py", "requirements.txt", "README.md"],
            "structure": ["src", "tests", "docs", "nis_system"]
        },
        "typescript": {
            "files": ["package.json", "tsconfig.json", "README.md"],
            "structure": ["src", "tests", "docs", "nis_system"]
        },
        "universal": {
            "files": ["README.md", "docker-compose.yml", ".gitignore"],
            "structure": ["agents", "config", "docs", "tests"]
        }
    }
    
    template_config = templates.get(language, templates["universal"])
    
    # Create directory structure
    for directory in template_config["structure"]:
        (project_path / directory).mkdir()
    
    # Create initial files
    for file_name in template_config["files"]:
        (project_path / file_name).touch()
    
    print(f"{Colors.GREEN}✅ Project '{project_name}' created successfully!{Colors.END}")
    print(f"{Colors.BLUE}📁 Next steps:{Colors.END}")
    print(f"  1. cd {project_name}")
    print(f"  2. nis-core integrate --type=comprehensive")
    print(f"  3. nis-core deploy --environment=local")

if __name__ == '__main__':
    # Handle async commands
    import inspect
    
    original_command = cli.command
    
    def async_command(*args, **kwargs):
        def decorator(f):
            if inspect.iscoroutinefunction(f):
                def wrapper(*args, **kwargs):
                    return asyncio.run(f(*args, **kwargs))
                wrapper.__name__ = f.__name__
                wrapper.__doc__ = f.__doc__
                return original_command(*args, **kwargs)(wrapper)
            else:
                return original_command(*args, **kwargs)(f)
        return decorator
    
    cli.command = async_command
    cli()
