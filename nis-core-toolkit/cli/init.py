#!/usr/bin/env python3
"""
NIS Core Toolkit - Project Initialization CLI
Enhanced `init` command for scaffolding new NIS Protocol projects
"""

import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
import shutil

console = Console()

def init_project(project_name: str):
    """
    Initialize a new NIS Protocol project with a best-practices structure.
    This command scaffolds a complete project, ready for development.
    """
    
    project_path = Path(project_name)
    
    if project_path.exists():
        console.print(f"❌ Project '{project_name}' already exists.", style="bold red")
        return False

    console.print(Panel.fit(
        f"[bold green]🚀 Initializing New NIS Project: {project_name}[/bold green]\n"
        "Scaffolding a best-practices project structure aligned with NIS Protocol v3.1.",
        title="NIS Project Initialization"
    ))

    try:
        # 1. Create root project directory
        project_path.mkdir()

        # 2. Create directory structure
        dirs_to_create = [
            "agents", "config", "data", "notebooks", "scripts",
            "src", "src/cognitive_agents", "src/core", "src/llm",
            "src/memory", "src/neural_hierarchy"
        ]
        for dir_name in dirs_to_create:
            (project_path / dir_name).mkdir()

        # Add .gitkeep to empty directories
        for empty_dir in ["agents", "data", "notebooks", "scripts"]:
            (project_path / empty_dir / ".gitkeep").touch()

        # 3. Create essential files from templates
        create_essential_files(project_path, project_name)

        # 4. Copy core NIS Protocol source code for alignment
        copy_nis_protocol_src(project_path)
        
        # 5. Display the generated structure
        display_project_tree(project_path)

        console.print(Panel(
            f"✅ [bold green]Project '{project_name}' initialized successfully![/bold green]\n\n"
            f"Navigate to your new project:\n"
            f"[cyan]cd {project_name}[/cyan]\n\n"
            f"Then, create your first agent:\n"
            f"[cyan]python ../nis-agent-toolkit/cli/main.py create reasoning my-first-agent[/cyan]",
            title="Next Steps"
        ))
        
        return True

    except Exception as e:
        console.print(f"❌ Error during project initialization: {e}", style="bold red")
        # Clean up partially created project
        if project_path.exists():
            shutil.rmtree(project_path)
        return False

def create_essential_files(project_path: Path, project_name: str):
    """Create essential project files with best-practices content."""
    
    files_to_create = {
        ".gitignore": GITIGNORE_TEMPLATE,
        "README.md": README_TEMPLATE.format(project_name=project_name),
        "requirements.txt": REQUIREMENTS_TEMPLATE,
        "config/agent_config.yml": CONFIG_TEMPLATE.format(name="default-agent", type="reasoning"),
        "config/system_config.yml": CONFIG_TEMPLATE.format(name=project_name, type="system"),
        "src/__init__.py": SRC_INIT_TEMPLATE.format(project_name=project_name)
    }

    for file_path, content in files_to_create.items():
        (project_path / file_path).write_text(content)

def copy_nis_protocol_src(project_path: Path):
    """Copy the core NIS Protocol source code for direct alignment."""
    
    # Path to the NIS Protocol source within the toolkit structure
    # This assumes the `init` command is run from the `NIS-TOOLKIT-SUIT` root
    nis_protocol_src_path = Path("nis-core-toolkit/src")

    if not nis_protocol_src_path.exists():
        console.print(f"⚠️ [yellow]Warning:[/yellow] Could not find NIS Protocol source at '{nis_protocol_src_path}'. Skipping source code copy.", style="dim")
        return

    # Copy key subdirectories into the new project's src
    core_logic_dirs = ["cognitive_agents", "core", "llm", "memory", "neural_hierarchy"]
    for subdir in core_logic_dirs:
        source_path = nis_protocol_src_path / subdir
        target_path = project_path / "src" / subdir
        if source_path.exists():
            # We already created the directories, so we copy the contents
            for item in source_path.iterdir():
                if item.is_file():
                    shutil.copy(item, target_path)
        else:
            console.print(f"⚠️ [yellow]Warning:[/yellow] Missing expected NIS Protocol directory: '{source_path}'", style="dim")


def display_project_tree(project_path: Path):
    """Display the generated project structure as a tree."""
    tree = Tree(f"🏗️  [bold green]{project_path.name}[/bold green]")
    
    def walk_directory(directory: Path, parent_branch):
        paths = sorted(
            list(directory.iterdir()),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        for path in paths:
            if path.name == '.gitkeep':
                continue
            if path.is_dir():
                branch = parent_branch.add(f"📂 [cyan]{path.name}[/cyan]")
                walk_directory(path, branch)
            else:
                parent_branch.add(f"📄 [green]{path.name}[/green]")

    walk_directory(project_path, tree)
    console.print(tree)

# ========================================
# TEMPLATES
# ========================================

GITIGNORE_TEMPLATE = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.idea/
.vscode/
*.swp
*.swo

# Data
data/
*.csv
*.json
*.db
*.sqlite3

# Logs
logs/
*.log
"""

README_TEMPLATE = """
# {project_name}

A new intelligent system powered by the NIS Protocol.

## 🚀 Getting Started

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure your system:**
    Edit the files in `config/`.

3.  **Create your first agent:**
    ```bash
    # Run from the NIS-TOOLKIT-SUIT root directory
    python nis-agent-toolkit/cli/main.py create reasoning my-first-agent
    ```

## 📝 Description

_Add a detailed description of your project here._
"""

REQUIREMENTS_TEMPLATE = """
# Core NIS Dependencies
# Add your project-specific dependencies here.

numpy
pandas
pyyaml
rich
click

# For Agent Development (from nis-agent-toolkit)
pydantic
asyncio-toolkit
typing-extensions

# For Integrity Checks (from nis-integrity-toolkit)
matplotlib
"""

CONFIG_TEMPLATE = """
# Configuration for {name}
# Type: {type}

version: "1.0.0"

parameters:
  # Add your configuration parameters here
  example_parameter: "example_value"
"""

SRC_INIT_TEMPLATE = '''
"""
{project_name} - A NIS Protocol Implementation
"""

__version__ = "0.1.0"
'''

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        init_project(sys.argv[1])
    else:
        console.print("Usage: python init.py <project_name>", style="bold red")
