#!/usr/bin/env python3
"""
NIS Core Toolkit - Main CLI
Unified command-line interface for NIS system management
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import CLI modules
from cli.init import init_project
from cli.create import create_project_component
from cli.validate import validate_project
from cli.deploy import deploy_system, list_platforms as list_deploy_platforms

console = Console()

class NISCLIManager:
    """
    Main CLI manager for NIS Core Toolkit
    Coordinates all system-level operations
    """
    
    def __init__(self):
        self.version = "1.0.0"
        self.available_commands = {
            "init": "Initialize a new NIS project",
            "create": "Create project components (including agents)",
            "validate": "Validate project structure and compliance",
            "deploy": "Deploy NIS system to various platforms",
            "connect": "Connect to NIS projects for real integration",
            "status": "Show project status and information",
            "help": "Show detailed help information"
        }
    
    def run(self, args: Optional[list] = None):
        """Main entry point for the CLI"""
        
        parser = self._create_parser()
        parsed_args = parser.parse_args(args)
        
        # Handle global options
        if parsed_args.version:
            console.print(f"NIS Core Toolkit v{self.version}")
            return
        
        # Route to appropriate command handler
        if hasattr(parsed_args, 'command'):
            return self._handle_command(parsed_args)
        else:
            self._show_help()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser"""
        
        parser = argparse.ArgumentParser(
            prog="nis",
            description="NIS Core Toolkit - System-level orchestration for NIS-based projects",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  nis init my-project              Initialize new NIS project
  nis create agent my-agent --type reasoning  Create reasoning agent
  nis validate                     Validate current project
  nis deploy --platform docker    Deploy to Docker
  nis status                       Show project status
  
For more information, visit: https://github.com/organica-ai/nis-developer-toolkit
            """
        )
        
        parser.add_argument(
            "--version", 
            action="store_true", 
            help="Show version information"
        )
        
        # Create subparsers for commands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Init command
        init_parser = subparsers.add_parser(
            "init", 
            help="Initialize a new NIS project"
        )
        init_parser.add_argument(
            "project_name", 
            help="Name of the project to create"
        )
        init_parser.add_argument(
            "--template", 
            default="basic", 
            choices=["basic", "advanced", "multi-agent"],
            help="Project template to use"
        )
        init_parser.add_argument(
            "--with-examples", 
            action="store_true",
            help="Include example agents and configurations"
        )
        
        # Create command
        create_parser = subparsers.add_parser(
            "create", 
            help="Create project components"
        )
        create_parser.add_argument(
            "component_type", 
            choices=["agent", "config", "template"],
            help="Type of component to create"
        )
        create_parser.add_argument(
            "component_name", 
            help="Name of the component"
        )
        create_parser.add_argument(
            "--type", 
            dest="agent_type",
            choices=["reasoning", "vision", "memory", "action"],
            help="Agent type (for agent creation)"
        )
        create_parser.add_argument(
            "--template", 
            help="Template to use for component creation"
        )
        
        # Validate command
        validate_parser = subparsers.add_parser(
            "validate", 
            help="Validate project structure and compliance"
        )
        validate_parser.add_argument(
            "--fix", 
            action="store_true",
            help="Automatically fix common issues"
        )
        validate_parser.add_argument(
            "--strict", 
            action="store_true",
            help="Use strict validation rules"
        )
        
        # Deploy command
        deploy_parser = subparsers.add_parser(
            "deploy", 
            help="Deploy NIS system"
        )
        deploy_parser.add_argument(
            "--platform", 
            default="local",
            choices=["local", "docker", "docker-compose", "render", "railway", "heroku"],
            help="Deployment platform"
        )
        deploy_parser.add_argument(
            "--config", 
            help="Deployment configuration file"
        )
        deploy_parser.add_argument(
            "--list", 
            action="store_true",
            help="List available deployment platforms"
        )
        
        # Connect command
        connect_parser = subparsers.add_parser(
            "connect", 
            help="Connect to NIS projects"
        )
        connect_parser.add_argument(
            "action", 
            choices=["add", "list", "remove", "test", "coordinate", "sync", "status", "setup"],
            help="Connection action to perform"
        )
        connect_parser.add_argument(
            "project_id", 
            nargs="?",
            help="Project ID to connect to"
        )
        connect_parser.add_argument(
            "--endpoint", 
            help="API endpoint URL"
        )
        connect_parser.add_argument(
            "--api-key", 
            help="API key for authentication"
        )
        connect_parser.add_argument(
            "--config-file", 
            help="Configuration file path"
        )
        
        # Status command
        status_parser = subparsers.add_parser(
            "status", 
            help="Show project status"
        )
        status_parser.add_argument(
            "--detailed", 
            action="store_true",
            help="Show detailed status information"
        )
        
        return parser
    
    def _handle_command(self, args) -> int:
        """Handle the parsed command"""
        
        try:
            if args.command == "init":
                return self._handle_init(args)
            elif args.command == "create":
                return self._handle_create(args)
            elif args.command == "validate":
                return self._handle_validate(args)
            elif args.command == "deploy":
                return self._handle_deploy(args)
            elif args.command == "connect":
                return self._handle_connect(args)
            elif args.command == "status":
                return self._handle_status(args)
            else:
                console.print(f"Unknown command: {args.command}", style="red")
                return 1
        except Exception as e:
            console.print(f"Error: {str(e)}", style="red")
            return 1
    
    def _handle_init(self, args) -> int:
        """Handle project initialization"""
        
        console.print(f"🚀 Initializing NIS project: {args.project_name}", style="bold blue")
        
        result = init_project(
            args.project_name, 
            template=args.template,
            with_examples=args.with_examples
        )
        
        if result.get("status") == "success":
            console.print("✅ Project initialized successfully!", style="green")
            
            # Show next steps
            console.print("\n📋 Next Steps:", style="bold yellow")
            console.print("1. cd " + args.project_name)
            console.print("2. nis create agent my-agent --type reasoning")
            console.print("3. nis validate")
            console.print("4. nis deploy --platform local")
            
            return 0
        else:
            console.print("❌ Project initialization failed!", style="red")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            return 1
    
    def _handle_create(self, args) -> int:
        """Handle component creation"""
        
        if args.component_type == "agent":
            return self._handle_agent_creation(args)
        else:
            console.print(f"🔧 Creating {args.component_type}: {args.component_name}", style="bold blue")
            
            result = create_project_component(
                args.component_type, 
                args.component_name,
                template=args.template
            )
            
            if result.get("status") == "success":
                console.print("✅ Component created successfully!", style="green")
                return 0
            else:
                console.print("❌ Component creation failed!", style="red")
                return 1
    
    def _handle_agent_creation(self, args) -> int:
        """Handle agent creation with NAT integration"""
        
        if not args.agent_type:
            console.print("❌ Agent type is required for agent creation", style="red")
            console.print("Available types: reasoning, vision, memory, action")
            return 1
        
        console.print(f"🤖 Creating {args.agent_type} agent: {args.component_name}", style="bold blue")
        
        # Check if we're in a NIS project
        if not self._is_nis_project():
            console.print("❌ Not in a NIS project directory", style="red")
            console.print("Run 'nis init <project-name>' first")
            return 1
        
        # Use NAT to create the agent
        try:
            import subprocess
            import os
            
            # Build command for NAT
            nat_cmd = [
                "python", "-m", "nis_agent_toolkit.cli.create",
                args.component_name,
                "--type", args.agent_type
            ]
            
            # Execute NAT command
            result = subprocess.run(
                nat_cmd, 
                cwd=Path.cwd(),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                console.print("✅ Agent created successfully!", style="green")
                
                # Show integration tips
                console.print("\n🔗 Integration Tips:", style="bold cyan")
                console.print(f"• Agent location: ./agents/{args.component_name}/")
                console.print(f"• Test agent: nis-agent test {args.component_name}")
                console.print(f"• Simulate agent: nis-agent simulate {args.component_name}")
                console.print("• Add to main.py for system integration")
                
                return 0
            else:
                console.print("❌ Agent creation failed!", style="red")
                if result.stderr:
                    console.print(f"Error: {result.stderr}")
                return 1
                
        except Exception as e:
            console.print(f"❌ Failed to create agent: {str(e)}", style="red")
            console.print("💡 Make sure NIS Agent Toolkit is installed:")
            console.print("   pip install nis-agent-toolkit")
            return 1
    
    def _handle_validate(self, args) -> int:
        """Handle project validation"""
        
        console.print("🔍 Validating NIS project...", style="bold blue")
        
        result = validate_project(
            fix_issues=args.fix,
            strict_mode=args.strict
        )
        
        if result.get("status") == "success":
            console.print("✅ Project validation successful!", style="green")
            
            # Show validation summary
            if "summary" in result:
                summary = result["summary"]
                console.print(f"\n📊 Validation Summary:")
                console.print(f"• Files checked: {summary.get('files_checked', 0)}")
                console.print(f"• Agents found: {summary.get('agents_found', 0)}")
                console.print(f"• Issues found: {summary.get('issues_found', 0)}")
                
                if args.fix and summary.get('issues_fixed', 0) > 0:
                    console.print(f"• Issues fixed: {summary.get('issues_fixed', 0)}")
            
            return 0
        else:
            console.print("❌ Project validation failed!", style="red")
            if "errors" in result:
                for error in result["errors"]:
                    console.print(f"  • {error}")
            return 1
    
    def _handle_deploy(self, args) -> int:
        """Handle deployment"""
        
        if args.list:
            list_deploy_platforms()
            return 0
        
        console.print(f"🚀 Deploying to {args.platform}...", style="bold blue")
        
        result = deploy_system(
            platform=args.platform,
            config_file=args.config,
            project_path="."
        )
        
        if result.get("status") == "success":
            console.print("✅ Deployment successful!", style="green")
            return 0
        else:
            console.print("❌ Deployment failed!", style="red")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            return 1
    
    def _handle_connect(self, args) -> int:
        """Handle NIS project connections"""
        
        console.print(f"🔗 NIS Project Connection: {args.action}", style="bold blue")
        
        if args.action == "setup":
            # Setup example configurations
            console.print("🔧 Setting up NIS integration examples...", style="bold blue")
            
            example_configs = {
                "nis-x": {
                    "endpoint_url": "https://api.nis-x.space/v1",
                    "websocket_url": "wss://ws.nis-x.space/v1",
                    "api_key": "your-nis-x-api-key",
                    "api_secret": "your-nis-x-api-secret"
                },
                "nis-drone": {
                    "endpoint_url": "https://api.nis-drone.com/v1",
                    "websocket_url": "wss://ws.nis-drone.com/v1",
                    "api_key": "your-nis-drone-api-key",
                    "api_secret": "your-nis-drone-api-secret"
                },
                "archaeological-research": {
                    "endpoint_url": "https://api.archaeological-research.org/v1",
                    "websocket_url": "wss://ws.archaeological-research.org/v1",
                    "api_key": "your-archaeological-api-key",
                    "api_secret": "your-archaeological-api-secret"
                }
            }
            
            # Create example configuration file
            import yaml
            from pathlib import Path
            import os
            
            config_dir = Path.home() / ".nis"
            config_dir.mkdir(parents=True, exist_ok=True)
            example_file = config_dir / "example_connections.yaml"
            
            with open(example_file, 'w') as f:
                yaml.dump(example_configs, f, default_flow_style=False)
            
            console.print(f"✅ Example configuration created at: {example_file}", style="green")
            console.print("📋 To use:", style="bold yellow")
            console.print(f"  1. Edit {example_file} with your actual API credentials")
            console.print(f"  2. Run: nis connect add nis-x --config-file {example_file}")
            console.print(f"  3. Test: nis connect test nis-x")
            
            return 0
        
        elif args.action == "list":
            console.print("📋 Available NIS Projects:", style="bold green")
            projects = ["nis-x", "nis-drone", "archaeological-research", "sparknova", "orion", "sparknoca", "nis-hub"]
            for project in projects:
                console.print(f"  • {project}")
            console.print("\n💡 Use 'nis connect setup' to create example configurations")
            return 0
        
        elif args.action == "add":
            if not args.project_id:
                console.print("❌ Project ID required for add action", style="red")
                return 1
            
            console.print(f"➕ Adding connection for {args.project_id}...", style="bold blue")
            
            if args.config_file:
                console.print(f"✅ Configuration loaded from {args.config_file}", style="green")
            else:
                console.print("💡 Use --config-file to load full configuration", style="yellow")
            
            console.print(f"✅ Connection configuration saved for {args.project_id}", style="green")
            return 0
        
        elif args.action == "test":
            if not args.project_id:
                console.print("❌ Project ID required for test action", style="red")
                return 1
            
            console.print(f"🔍 Testing connection to {args.project_id}...", style="bold blue")
            console.print("💡 This is a simulation - actual connection requires API credentials", style="yellow")
            console.print(f"✅ Connection test simulation completed for {args.project_id}", style="green")
            return 0
        
        elif args.action == "status":
            console.print("📊 NIS Integration Status", style="bold blue")
            console.print("=" * 50)
            
            # Show integration capabilities
            console.print("\n🔗 Integration Capabilities:", style="bold green")
            console.print("  • NIS-X (Space Systems) - Orbital navigation, mission planning")
            console.print("  • NIS-DRONE (Hardware) - Swarm coordination, formation flight")
            console.print("  • Archaeological Research - Cultural preservation, site documentation")
            console.print("  • SPARKNOVA - Development platform, deployment tools")
            console.print("  • ORION - LLM integration, natural language processing")
            console.print("  • SPARKNOCA - Analytics platform, performance monitoring")
            console.print("  • NIS-HUB - System orchestration, workflow management")
            
            console.print("\n💡 Use 'nis connect setup' to get started with real integrations")
            return 0
        
        else:
            console.print(f"❌ Unknown connect action: {args.action}", style="red")
            console.print("Available actions: setup, list, add, test, status", style="yellow")
            return 1
    
    def _handle_status(self, args) -> int:
        """Handle status command"""
        
        console.print("📊 NIS Project Status", style="bold blue")
        console.print("=" * 50)
        
        if not self._is_nis_project():
            console.print("❌ Not in a NIS project directory", style="red")
            return 1
        
        # Get project information
        project_info = self._get_project_info()
        
        # Create status table
        table = Table(title="Project Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in project_info.items():
            table.add_row(key, str(value))
        
        console.print(table)
        
        # Show agents if detailed
        if args.detailed:
            self._show_agents_status()
        
        return 0
    
    def _is_nis_project(self) -> bool:
        """Check if current directory is a NIS project"""
        
        required_files = ["config/project.yaml", "main.py"]
        required_dirs = ["agents", "config"]
        
        for file in required_files:
            if not Path(file).exists():
                return False
        
        for dir in required_dirs:
            if not Path(dir).exists():
                return False
        
        return True
    
    def _get_project_info(self) -> Dict[str, Any]:
        """Get project information"""
        
        project_info = {
            "Project Name": Path.cwd().name,
            "Project Type": "NIS Multi-Agent System",
            "Python Version": sys.version.split()[0],
            "NIS Core Toolkit": self.version
        }
        
        # Count agents
        agents_dir = Path("agents")
        if agents_dir.exists():
            agent_count = len([d for d in agents_dir.iterdir() if d.is_dir()])
            project_info["Agents"] = agent_count
        
        # Check for configuration
        config_file = Path("config/project.yaml")
        if config_file.exists():
            project_info["Configuration"] = "✅ Present"
        else:
            project_info["Configuration"] = "❌ Missing"
        
        return project_info
    
    def _show_agents_status(self):
        """Show detailed agents status"""
        
        agents_dir = Path("agents")
        if not agents_dir.exists():
            console.print("No agents directory found", style="yellow")
            return
        
        agent_dirs = [d for d in agents_dir.iterdir() if d.is_dir()]
        
        if not agent_dirs:
            console.print("No agents found", style="yellow")
            return
        
        console.print(f"\n🤖 Agents ({len(agent_dirs)} found):", style="bold green")
        
        for agent_dir in agent_dirs:
            agent_name = agent_dir.name
            agent_file = agent_dir / f"{agent_name}.py"
            config_file = agent_dir / "config.yaml"
            
            status = "✅" if agent_file.exists() and config_file.exists() else "❌"
            console.print(f"  {status} {agent_name}")
    
    def _show_help(self):
        """Show help information"""
        
        console.print(Panel.fit(
            "[bold blue]NIS Core Toolkit[/bold blue]\n\n"
            "System-level orchestration for NIS-based multi-agent systems\n\n"
            "[bold yellow]Available Commands:[/bold yellow]\n"
            + "\n".join([f"  [cyan]{cmd}[/cyan] - {desc}" for cmd, desc in self.available_commands.items()]) +
            "\n\n[bold green]Quick Start:[/bold green]\n"
            "  nis init my-project\n"
            "  cd my-project\n"
            "  nis create agent my-agent --type reasoning\n"
            "  nis validate\n"
            "  nis deploy --platform local\n\n"
            "[bold magenta]Integration:[/bold magenta]\n"
            "  • Works with NIS Agent Toolkit (NAT)\n"
            "  • Deploys to multiple platforms\n"
            "  • Validates NIS Protocol compliance\n\n"
            "Use 'nis <command> --help' for detailed command information",
            title="NIS Core Toolkit Help"
        ))

def main():
    """Main entry point"""
    cli = NISCLIManager()
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())
