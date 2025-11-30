"""
NIS CLI Create Command
Creates new NIS projects, components, and scaffolds
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any

from .base import BaseCommand
from ..utils.logger import success, error, info, step, header
from ..utils.config import NISConfig, create_default_config

class CreateCommand(BaseCommand):
    """Create new NIS projects and components"""
    
    @classmethod
    def register(cls, subparsers):
        """Register the create command"""
        parser = subparsers.add_parser(
            'create',
            help='Create new NIS project or component',
            description='Create new NIS projects, agents, or components from templates'
        )
        
        subcommands = parser.add_subparsers(
            dest='create_type',
            help='What to create',
            metavar='<type>'
        )
        
        # Project creation
        project_parser = subcommands.add_parser(
            'project',
            help='Create a new NIS project'
        )
        project_parser.add_argument(
            'name',
            help='Project name'
        )
        project_parser.add_argument(
            '--template', '-t',
            choices=['basic', 'full', 'edge', 'research'],
            default='basic',
            help='Project template to use'
        )
        project_parser.add_argument(
            '--no-docker',
            action='store_true',
            help='Skip Docker configuration'
        )
        project_parser.add_argument(
            '--no-git',
            action='store_true', 
            help='Skip Git initialization'
        )
        
        # Agent creation
        agent_parser = subcommands.add_parser(
            'agent',
            help='Create a new NIS agent'
        )
        agent_parser.add_argument(
            'name',
            help='Agent name'
        )
        agent_parser.add_argument(
            '--type', '-t',
            choices=['conscious', 'reasoning', 'memory', 'perception', 'action'],
            default='conscious',
            help='Agent type to create'
        )
        
        # Component creation
        component_parser = subcommands.add_parser(
            'component',
            help='Create a new NIS component'
        )
        component_parser.add_argument(
            'name',
            help='Component name'
        )
        component_parser.add_argument(
            '--type', '-t',
            choices=['provider', 'adapter', 'service', 'utility'],
            default='service',
            help='Component type to create'
        )
    
    def execute(self, args) -> int:
        """Execute the create command"""
        header("🏗️  NIS TOOLKIT SUIT - Project Creator")
        
        if not args.create_type:
            error("Please specify what to create (project, agent, component)")
            return 1
        
        try:
            if args.create_type == 'project':
                return self._create_project(args)
            elif args.create_type == 'agent':
                return self._create_agent(args)
            elif args.create_type == 'component':
                return self._create_component(args)
            else:
                error(f"Unknown create type: {args.create_type}")
                return 1
                
        except Exception as e:
            error(f"Creation failed: {e}")
            return 1
    
    def _create_project(self, args) -> int:
        """Create a new NIS project"""
        project_name = args.name
        template = args.template
        
        # Validate project name
        if not self._validate_project_name(project_name):
            return 1
        
        project_path = Path(project_name)
        
        # Check if directory exists
        if project_path.exists():
            error(f"Directory already exists: {project_name}")
            return 1
        
        step(f"Creating NIS project: {project_name}")
        step(f"Using template: {template}")
        
        # Create project structure
        self._create_project_structure(project_path, template)
        
        # Generate configuration
        self._create_project_config(project_path, project_name, template)
        
        # Create template files
        self._create_template_files(project_path, project_name, template)
        
        # Initialize Docker if requested
        if not args.no_docker:
            self._setup_docker(project_path, template)
        
        # Initialize Git if requested
        if not args.no_git:
            self._setup_git(project_path)
        
        # Create README and documentation
        self._create_documentation(project_path, project_name, template)
        
        success(f"✨ Created NIS project: {project_name}")
        info(f"📁 Project directory: {project_path.absolute()}")
        
        # Show next steps
        self._show_next_steps(project_name, template)
        
        return 0
    
    def _validate_project_name(self, name: str) -> bool:
        """Validate project name"""
        if not name:
            error("Project name cannot be empty")
            return False
        
        if not name.replace('-', '').replace('_', '').isalnum():
            error("Project name can only contain letters, numbers, hyphens, and underscores")
            return False
        
        if name.startswith('-') or name.startswith('_'):
            error("Project name cannot start with a hyphen or underscore")
            return False
        
        return True
    
    def _create_project_structure(self, project_path: Path, template: str):
        """Create the project directory structure"""
        
        # Base structure for all templates
        directories = [
            "src",
            "src/agents",
            "src/agents/consciousness",
            "src/agents/reasoning", 
            "src/agents/memory",
            "src/core",
            "src/utils",
            "tests",
            "tests/unit",
            "tests/integration",
            "docs",
            "examples",
            "configs"
        ]
        
        # Template-specific additions
        if template == 'full':
            directories.extend([
                "src/agents/perception",
                "src/agents/action",
                "src/agents/learning",
                "src/infrastructure", 
                "src/monitoring",
                "deployment",
                "deployment/kubernetes",
                "scripts"
            ])
        elif template == 'edge':
            directories.extend([
                "src/edge",
                "src/lightweight",
                "deployment/edge"
            ])
        elif template == 'research':
            directories.extend([
                "src/experiments",
                "src/research",
                "notebooks",
                "data",
                "results"
            ])
        
        # Create all directories
        for directory in directories:
            self.create_directory(project_path / directory, f"directory '{directory}'")
    
    def _create_project_config(self, project_path: Path, project_name: str, template: str):
        """Create project configuration files"""
        
        # Create NIS configuration
        config = NISConfig()
        config.project_name = project_name
        config.project_description = f"NIS Protocol project created with {template} template"
        
        # Template-specific configurations
        if template == 'edge':
            config.enable_edge_computing = True
            config.docker_tag = f"{project_name}-edge"
        elif template == 'research':
            config.debug = True
            config.enable_monitoring = False
        elif template == 'full':
            config.enable_monitoring = True
            config.enable_consciousness = True
            config.enable_provider_router = True
        
        # Save configuration
        config_path = project_path / "nis.config.yaml"
        with open(config_path, 'w') as f:
            import yaml
            config_dict = {field.name: getattr(config, field.name) 
                          for field in config.__dataclass_fields__.values()}
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        info(f"Created configuration: nis.config.yaml")
    
    def _create_template_files(self, project_path: Path, project_name: str, template: str):
        """Create template-specific files"""
        
        replacements = {
            'PROJECT_NAME': project_name,
            'PROJECT_NAME_UPPER': project_name.upper().replace('-', '_'),
            'PROJECT_NAME_LOWER': project_name.lower().replace('-', '_'),
            'TEMPLATE': template,
            'NIS_VERSION': self.config.nis_version
        }
        
        # Main application file
        main_content = self._get_main_template(template, replacements)
        (project_path / "main.py").write_text(main_content)
        
        # Requirements file
        requirements_content = self._get_requirements_template(template)
        (project_path / "requirements.txt").write_text(requirements_content)
        
        # Example agent
        agent_content = self._get_agent_template(template, replacements)
        (project_path / "src" / "agents" / "example_agent.py").write_text(agent_content)
        
        # Tests
        test_content = self._get_test_template(template, replacements)
        (project_path / "tests" / "test_example.py").write_text(test_content)
    
    def _setup_docker(self, project_path: Path, template: str):
        """Setup Docker configuration"""
        step("Setting up Docker configuration")
        
        # Copy Dockerfile from templates
        dockerfile_content = self._get_dockerfile_template(template)
        (project_path / "Dockerfile").write_text(dockerfile_content)
        
        # Copy docker-compose
        compose_content = self._get_compose_template(template)
        (project_path / "docker-compose.yml").write_text(compose_content)
        
        # Create .dockerignore
        dockerignore_content = self._get_dockerignore_template()
        (project_path / ".dockerignore").write_text(dockerignore_content)
        
        info("Docker configuration created")
    
    def _setup_git(self, project_path: Path):
        """Initialize Git repository"""
        step("Initializing Git repository")
        
        try:
            # Initialize git
            self.run_command(['git', 'init'], cwd=str(project_path))
            
            # Create .gitignore
            gitignore_content = self._get_gitignore_template()
            (project_path / ".gitignore").write_text(gitignore_content)
            
            # Initial commit
            self.run_command(['git', 'add', '.'], cwd=str(project_path))
            self.run_command([
                'git', 'commit', '-m', 'Initial commit: NIS project created'
            ], cwd=str(project_path))
            
            success("Git repository initialized")
            
        except Exception as e:
            error(f"Failed to initialize Git: {e}")
    
    def _create_documentation(self, project_path: Path, project_name: str, template: str):
        """Create project documentation"""
        
        # README.md
        readme_content = f"""# {project_name}

NIS Protocol project created with {template} template.

## 🚀 Quick Start

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Docker
```bash
# Build and run with Docker
docker build -t {project_name} .
docker run -p 8000:8000 {project_name}
```

### NIS CLI
```bash
# Deploy with NIS CLI
nis deploy docker

# Run tests
nis test

# Monitor application
nis monitor
```

## 📁 Project Structure

- `src/` - Source code
- `tests/` - Test files  
- `docs/` - Documentation
- `examples/` - Example code
- `configs/` - Configuration files

## 🧠 NIS Features

This project includes:
- ✅ NIS Protocol v{self.config.nis_version}
- ✅ Enhanced Consciousness Agent
- ✅ Provider Router
- ✅ Security Hardened Dependencies
- ✅ Docker Support
- ✅ Monitoring & Observability

## 📚 Learn More

- [NIS Protocol Documentation](https://github.com/nis-protocol)
- [NIS Toolkit Guide](./docs/guide.md)
- [API Reference](./docs/api.md)
"""
        
        (project_path / "README.md").write_text(readme_content)
        info("Created README.md")
    
    def _show_next_steps(self, project_name: str, template: str):
        """Show next steps to the user"""
        header("🎯 Next Steps")
        
        print(f"📁 cd {project_name}")
        print("📦 pip install -r requirements.txt")
        print("🚀 python main.py")
        print("")
        print("For development:")
        print("🐳 nis deploy docker --dev")
        print("🧪 nis test")
        print("📊 nis monitor")
    
    def _create_agent(self, args) -> int:
        """Create a new agent"""
        # Ensure we're in a project
        project_root = self.ensure_project_root()
        
        agent_name = args.name
        agent_type = args.type
        
        step(f"Creating {agent_type} agent: {agent_name}")
        
        # Create agent file
        agent_path = project_root / "src" / "agents" / f"{agent_name}_agent.py"
        agent_content = self._get_agent_class_template(agent_name, agent_type)
        
        agent_path.write_text(agent_content)
        success(f"Created agent: {agent_path}")
        
        return 0
    
    def _create_component(self, args) -> int:
        """Create a new component"""
        # Ensure we're in a project
        project_root = self.ensure_project_root()
        
        component_name = args.name
        component_type = args.type
        
        step(f"Creating {component_type} component: {component_name}")
        
        # Create component file
        component_path = project_root / "src" / f"{component_name}_{component_type}.py"
        component_content = self._get_component_template(component_name, component_type)
        
        component_path.write_text(component_content)
        success(f"Created component: {component_path}")
        
        return 0
    
    # Template content methods
    def _get_main_template(self, template: str, replacements: Dict[str, str]) -> str:
        """Get main.py template content"""
        return f'''#!/usr/bin/env python3
"""
{replacements["PROJECT_NAME"]} - NIS Protocol Application
Generated with {template} template
"""

import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main application entry point"""
    logger.info("Starting {replacements['PROJECT_NAME']} v1.0.0")
    
    try:
        # Initialize NIS components
        from src.agents.example_agent import ExampleAgent
        
        # Create and initialize agent
        agent = ExampleAgent()
        await agent.initialize()
        
        # Run agent
        result = await agent.process("Hello, NIS Protocol!")
        logger.info(f"Agent response: {{result}}")
        
    except Exception as e:
        logger.error(f"Application error: {{e}}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
'''
    
    def _get_requirements_template(self, template: str) -> str:
        """Get requirements.txt template content"""
        base_requirements = """# NIS TOOLKIT SUIT v4.0.0 - Core Dependencies
fastapi>=0.116.0,<0.120.0
uvicorn[standard]>=0.29.0,<0.35.0
pydantic>=2.9.0,<3.0.0
numpy>=1.24.0,<2.0.0
requests>=2.32.0,<3.0.0
"""
        
        if template == 'full':
            base_requirements += """
# Full template additional dependencies
transformers>=4.53.0,<5.0.0
torch>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0
redis>=4.5.0,<5.0.0
"""
        elif template == 'research':
            base_requirements += """
# Research template dependencies
jupyter>=1.0.0
matplotlib>=3.6.0
pandas>=2.0.0,<3.0.0
plotly>=5.15.0,<6.0.0
"""
        
        return base_requirements
    
    def _get_agent_template(self, template: str, replacements: Dict[str, str]) -> str:
        """Get example agent template"""
        return f'''"""
Example NIS Agent for {replacements["PROJECT_NAME"]}
"""

import logging
from typing import Dict, Any

class ExampleAgent:
    """Example NIS Protocol agent"""
    
    def __init__(self):
        self.agent_id = "example_agent"
        self.logger = logging.getLogger(self.agent_id)
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        self.logger.info("Initializing Example Agent")
        return True
    
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data"""
        self.logger.info(f"Processing: {{input_data}}")
        
        # Simple processing logic
        result = {{
            "agent_id": self.agent_id,
            "input": input_data,
            "output": f"Processed: {{input_data}}",
            "status": "success"
        }}
        
        return result
'''
    
    def _get_test_template(self, template: str, replacements: Dict[str, str]) -> str:
        """Get test template"""
        return f'''"""
Tests for {replacements["PROJECT_NAME"]}
"""

import pytest
import asyncio
from src.agents.example_agent import ExampleAgent

@pytest.mark.asyncio
async def test_example_agent():
    """Test the example agent"""
    agent = ExampleAgent()
    
    # Initialize
    assert await agent.initialize() == True
    
    # Process data
    result = await agent.process("test input")
    
    assert result["status"] == "success"
    assert "test input" in result["output"]

def test_agent_creation():
    """Test agent can be created"""
    agent = ExampleAgent()
    assert agent.agent_id == "example_agent"
'''
    
    def _get_dockerfile_template(self, template: str) -> str:
        """Get Dockerfile template"""
        return """# NIS Project Dockerfile
FROM python:3.11-slim AS base

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
"""
    
    def _get_compose_template(self, template: str) -> str:
        """Get docker-compose.yml template"""
        return """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NIS_ENVIRONMENT=docker
    volumes:
      - ./logs:/app/logs
"""
    
    def _get_dockerignore_template(self) -> str:
        """Get .dockerignore template"""
        return """__pycache__
*.pyc
*.pyo
.git
.pytest_cache
.coverage
*.log
docs/
tests/
"""
    
    def _get_gitignore_template(self) -> str:
        """Get .gitignore template"""
        return """# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
.env

# IDE
.vscode/
.idea/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# NIS specific
nis.config.local.yaml
.nis-cache/
"""
    
    def _get_agent_class_template(self, name: str, agent_type: str) -> str:
        """Get agent class template"""
        return f'''"""
{name.title()} {agent_type.title()} Agent
"""

import logging
from typing import Dict, Any

class {name.title()}Agent:
    """NIS Protocol {agent_type} agent"""
    
    def __init__(self):
        self.agent_id = "{name}_{agent_type}_agent"
        self.agent_type = "{agent_type}"
        self.logger = logging.getLogger(self.agent_id)
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        self.logger.info(f"Initializing {{self.agent_id}}")
        return True
    
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input through {agent_type} capabilities"""
        self.logger.info(f"{{self.agent_type.title()}} processing: {{input_data}}")
        
        # Implement {agent_type}-specific logic here
        result = {{
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "input": input_data,
            "output": f"{agent_type.title()} result for: {{input_data}}",
            "confidence": 0.85,
            "status": "success"
        }}
        
        return result
'''
    
    def _get_component_template(self, name: str, component_type: str) -> str:
        """Get component template"""
        return f'''"""
{name.title()} {component_type.title()} Component
"""

import logging
from typing import Dict, Any, Optional

class {name.title()}{component_type.title()}:
    """NIS Protocol {component_type} component"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.component_id = "{name}_{component_type}"
        self.component_type = "{component_type}"
        self.config = config or {{}}
        self.logger = logging.getLogger(self.component_id)
    
    def initialize(self) -> bool:
        """Initialize the component"""
        self.logger.info(f"Initializing {{self.component_id}}")
        return True
    
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute component functionality"""
        self.logger.info(f"Executing {{self.component_type}} component")
        
        result = {{
            "component_id": self.component_id,
            "component_type": self.component_type,
            "status": "success"
        }}
        
        return result
'''
