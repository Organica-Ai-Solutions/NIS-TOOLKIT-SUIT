#!/usr/bin/env python3
"""
NIS Core Toolkit - Deployment Module
Deploy NIS systems locally or to cloud platforms
"""

import os
import json
import yaml
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

console = Console()

class NISDeployer:
    """
    NIS system deployment manager
    Honest deployment - practical deployment options, not hype
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(".")
        self.deployment_configs = {}
        self.supported_platforms = {
            "local": "Local development deployment",
            "docker": "Docker container deployment",
            "docker-compose": "Multi-container Docker deployment",
            "render": "Render.com cloud deployment",
            "railway": "Railway.app deployment",
            "heroku": "Heroku deployment"
        }
    
    def deploy_system(self, platform: str = "local", config_file: str = None) -> Dict[str, Any]:
        """Deploy NIS system to specified platform"""
        
        console.print(f"🚀 Deploying NIS system to {platform}...", style="bold blue")
        
        # Load deployment configuration
        deploy_config = self._load_deployment_config(config_file)
        
        # Validate project before deployment
        if not self._validate_project():
            return {"status": "failed", "error": "Project validation failed"}
        
        # Platform-specific deployment
        if platform == "local":
            result = self._deploy_local(deploy_config)
        elif platform == "docker":
            result = self._deploy_docker(deploy_config)
        elif platform == "docker-compose":
            result = self._deploy_docker_compose(deploy_config)
        elif platform == "render":
            result = self._deploy_render(deploy_config)
        elif platform == "railway":
            result = self._deploy_railway(deploy_config)
        elif platform == "heroku":
            result = self._deploy_heroku(deploy_config)
        else:
            return {"status": "failed", "error": f"Unsupported platform: {platform}"}
        
        # Display deployment results
        self._display_deployment_results(result, platform)
        
        return result
    
    def _load_deployment_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load deployment configuration"""
        
        if config_file:
            config_path = Path(config_file)
        else:
            config_path = self.project_root / "config" / "deploy.yaml"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                console.print(f"Warning: Could not load deployment config: {e}", style="yellow")
        
        # Default configuration
        return {
            "deployment": {
                "port": 8000,
                "host": "0.0.0.0",
                "workers": 1,
                "timeout": 30
            },
            "environment": {
                "NIS_ENV": "production",
                "NIS_LOG_LEVEL": "INFO"
            },
            "resources": {
                "memory": "512MB",
                "cpu": "0.5"
            }
        }
    
    def _validate_project(self) -> bool:
        """Validate project is ready for deployment"""
        
        required_files = ["main.py", "requirements.txt"]
        required_dirs = ["agents", "config"]
        
        for file in required_files:
            if not (self.project_root / file).exists():
                console.print(f"❌ Missing required file: {file}", style="red")
                return False
        
        for dir in required_dirs:
            if not (self.project_root / dir).exists():
                console.print(f"❌ Missing required directory: {dir}", style="red")
                return False
        
        return True
    
    def _deploy_local(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy system locally"""
        
        console.print("🏠 Starting local deployment...", style="blue")
        
        try:
            # Install dependencies
            console.print("📦 Installing dependencies...")
            subprocess.run([
                "pip", "install", "-r", "requirements.txt"
            ], cwd=self.project_root, check=True, capture_output=True)
            
            # Create deployment script
            deploy_script = self._create_local_deployment_script(config)
            
            # Start the system
            console.print("🔥 Starting NIS system...")
            
            return {
                "status": "success",
                "platform": "local",
                "url": f"http://localhost:{config['deployment']['port']}",
                "deployment_script": deploy_script,
                "environment": config.get("environment", {}),
                "instructions": [
                    f"cd {self.project_root}",
                    f"python {deploy_script}",
                    "System will be available at the provided URL"
                ]
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "platform": "local",
                "error": str(e)
            }
    
    def _deploy_docker(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy system using Docker"""
        
        console.print("🐳 Starting Docker deployment...", style="blue")
        
        try:
            # Create Dockerfile
            dockerfile = self._create_dockerfile(config)
            
            # Build Docker image
            console.print("🔨 Building Docker image...")
            image_name = f"nis-{self.project_root.name.lower()}"
            
            build_result = subprocess.run([
                "docker", "build", "-t", image_name, "."
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if build_result.returncode != 0:
                return {
                    "status": "failed",
                    "platform": "docker",
                    "error": f"Docker build failed: {build_result.stderr}"
                }
            
            # Create run script
            run_script = self._create_docker_run_script(image_name, config)
            
            return {
                "status": "success",
                "platform": "docker",
                "image_name": image_name,
                "dockerfile": dockerfile,
                "run_script": run_script,
                "instructions": [
                    f"Docker image '{image_name}' built successfully",
                    f"Run: {run_script}",
                    f"Access at: http://localhost:{config['deployment']['port']}"
                ]
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "platform": "docker",
                "error": str(e)
            }
    
    def _deploy_docker_compose(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy system using Docker Compose"""
        
        console.print("🐳 Starting Docker Compose deployment...", style="blue")
        
        try:
            # Create docker-compose.yml
            compose_file = self._create_docker_compose_file(config)
            
            # Create supporting files
            dockerfile = self._create_dockerfile(config)
            env_file = self._create_env_file(config)
            
            return {
                "status": "success",
                "platform": "docker-compose",
                "compose_file": compose_file,
                "dockerfile": dockerfile,
                "env_file": env_file,
                "instructions": [
                    "docker-compose up -d",
                    "docker-compose logs -f",
                    f"Access at: http://localhost:{config['deployment']['port']}"
                ]
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "platform": "docker-compose",
                "error": str(e)
            }
    
    def _deploy_render(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy system to Render.com"""
        
        console.print("🌐 Preparing Render.com deployment...", style="blue")
        
        try:
            # Create render.yaml
            render_config = self._create_render_config(config)
            
            # Create startup script
            startup_script = self._create_render_startup_script(config)
            
            return {
                "status": "success",
                "platform": "render",
                "render_config": render_config,
                "startup_script": startup_script,
                "instructions": [
                    "1. Push your code to a GitHub repository",
                    "2. Connect your repository to Render.com",
                    "3. Use the generated render.yaml for configuration",
                    "4. Deploy using the Render dashboard"
                ]
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "platform": "render",
                "error": str(e)
            }
    
    def _deploy_railway(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy system to Railway.app"""
        
        console.print("🚂 Preparing Railway.app deployment...", style="blue")
        
        try:
            # Create railway.json
            railway_config = self._create_railway_config(config)
            
            # Create Procfile
            procfile = self._create_procfile(config)
            
            return {
                "status": "success",
                "platform": "railway",
                "railway_config": railway_config,
                "procfile": procfile,
                "instructions": [
                    "1. Install Railway CLI: npm install -g @railway/cli",
                    "2. Login: railway login",
                    "3. Deploy: railway up",
                    "4. Set environment variables in Railway dashboard"
                ]
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "platform": "railway",
                "error": str(e)
            }
    
    def _deploy_heroku(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy system to Heroku"""
        
        console.print("🌪️  Preparing Heroku deployment...", style="blue")
        
        try:
            # Create Procfile
            procfile = self._create_procfile(config)
            
            # Create app.json
            app_config = self._create_heroku_app_config(config)
            
            return {
                "status": "success",
                "platform": "heroku",
                "procfile": procfile,
                "app_config": app_config,
                "instructions": [
                    "1. Install Heroku CLI",
                    "2. heroku login",
                    "3. heroku create your-app-name",
                    "4. git push heroku main",
                    "5. heroku config:set environment variables"
                ]
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "platform": "heroku",
                "error": str(e)
            }
    
    def _create_local_deployment_script(self, config: Dict[str, Any]) -> str:
        """Create local deployment script"""
        
        script_content = f"""#!/usr/bin/env python3
# NIS System Local Deployment Script
# Generated by NIS Core Toolkit

import os
import sys
from pathlib import Path

# Set environment variables
os.environ['NIS_ENV'] = '{config.get('environment', {}).get('NIS_ENV', 'production')}'
os.environ['NIS_LOG_LEVEL'] = '{config.get('environment', {}).get('NIS_LOG_LEVEL', 'INFO')}'
os.environ['NIS_PORT'] = '{config['deployment']['port']}'
os.environ['NIS_HOST'] = '{config['deployment']['host']}'

# Start the NIS system
if __name__ == "__main__":
    print("🚀 Starting NIS System...")
    print(f"Environment: {{os.environ.get('NIS_ENV')}}")
    print(f"Port: {{os.environ.get('NIS_PORT')}}")
    print(f"Host: {{os.environ.get('NIS_HOST')}}")
    
    # Import and run main
    try:
        from main import main
        main()
    except ImportError:
        print("Error: Could not import main module")
        sys.exit(1)
"""
        
        script_path = self.project_root / "deploy.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return "deploy.py"
    
    def _create_dockerfile(self, config: Dict[str, Any]) -> str:
        """Create Dockerfile"""
        
        dockerfile_content = f"""# NIS System Dockerfile
# Generated by NIS Core Toolkit

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV NIS_ENV={config.get('environment', {}).get('NIS_ENV', 'production')}
ENV NIS_LOG_LEVEL={config.get('environment', {}).get('NIS_LOG_LEVEL', 'INFO')}
ENV NIS_PORT={config['deployment']['port']}
ENV NIS_HOST={config['deployment']['host']}

# Expose port
EXPOSE {config['deployment']['port']}

# Run the application
CMD ["python", "main.py"]
"""
        
        dockerfile_path = self.project_root / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        return "Dockerfile"
    
    def _create_docker_compose_file(self, config: Dict[str, Any]) -> str:
        """Create docker-compose.yml"""
        
        compose_content = f"""# NIS System Docker Compose
# Generated by NIS Core Toolkit

version: '3.8'

services:
  nis-system:
    build: .
    ports:
      - "{config['deployment']['port']}:{config['deployment']['port']}"
    environment:
      - NIS_ENV={config.get('environment', {}).get('NIS_ENV', 'production')}
      - NIS_LOG_LEVEL={config.get('environment', {}).get('NIS_LOG_LEVEL', 'INFO')}
      - NIS_PORT={config['deployment']['port']}
      - NIS_HOST={config['deployment']['host']}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{config['deployment']['port']}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add Redis for memory backend
  # redis:
  #   image: redis:7-alpine
  #   ports:
  #     - "6379:6379"
  #   volumes:
  #     - redis_data:/data
  #   restart: unless-stopped

# volumes:
#   redis_data:
"""
        
        compose_path = self.project_root / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        return "docker-compose.yml"
    
    def _create_render_config(self, config: Dict[str, Any]) -> str:
        """Create render.yaml"""
        
        render_content = f"""# NIS System Render Configuration
# Generated by NIS Core Toolkit

services:
  - type: web
    name: nis-system
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py
    envVars:
      - key: NIS_ENV
        value: {config.get('environment', {}).get('NIS_ENV', 'production')}
      - key: NIS_LOG_LEVEL
        value: {config.get('environment', {}).get('NIS_LOG_LEVEL', 'INFO')}
      - key: NIS_PORT
        value: {config['deployment']['port']}
      - key: NIS_HOST
        value: {config['deployment']['host']}
    plan: free
    # healthCheckPath: /health
"""
        
        render_path = self.project_root / "render.yaml"
        with open(render_path, 'w') as f:
            f.write(render_content)
        
        return "render.yaml"
    
    def _create_procfile(self, config: Dict[str, Any]) -> str:
        """Create Procfile"""
        
        procfile_content = f"web: python main.py"
        
        procfile_path = self.project_root / "Procfile"
        with open(procfile_path, 'w') as f:
            f.write(procfile_content)
        
        return "Procfile"
    
    def _create_env_file(self, config: Dict[str, Any]) -> str:
        """Create .env file"""
        
        env_content = f"""# NIS System Environment Variables
# Generated by NIS Core Toolkit

NIS_ENV={config.get('environment', {}).get('NIS_ENV', 'production')}
NIS_LOG_LEVEL={config.get('environment', {}).get('NIS_LOG_LEVEL', 'INFO')}
NIS_PORT={config['deployment']['port']}
NIS_HOST={config['deployment']['host']}
"""
        
        env_path = self.project_root / ".env"
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        return ".env"
    
    def _create_railway_config(self, config: Dict[str, Any]) -> str:
        """Create railway.json"""
        
        railway_config = {
            "build": {
                "builder": "NIXPACKS"
            },
            "deploy": {
                "startCommand": "python main.py",
                "healthcheckPath": "/health",
                "healthcheckTimeout": 300
            }
        }
        
        railway_path = self.project_root / "railway.json"
        with open(railway_path, 'w') as f:
            json.dump(railway_config, f, indent=2)
        
        return "railway.json"
    
    def _create_heroku_app_config(self, config: Dict[str, Any]) -> str:
        """Create app.json for Heroku"""
        
        app_config = {
            "name": f"nis-{self.project_root.name.lower()}",
            "description": "NIS Protocol-based multi-agent system",
            "repository": "https://github.com/your-username/your-repo",
            "keywords": ["nis", "ai", "agents", "python"],
            "env": {
                "NIS_ENV": {
                    "description": "NIS environment",
                    "value": config.get('environment', {}).get('NIS_ENV', 'production')
                },
                "NIS_LOG_LEVEL": {
                    "description": "Logging level",
                    "value": config.get('environment', {}).get('NIS_LOG_LEVEL', 'INFO')
                }
            },
            "formation": {
                "web": {
                    "quantity": 1,
                    "size": "free"
                }
            },
            "buildpacks": [
                {
                    "url": "heroku/python"
                }
            ]
        }
        
        app_path = self.project_root / "app.json"
        with open(app_path, 'w') as f:
            json.dump(app_config, f, indent=2)
        
        return "app.json"
    
    def _create_render_startup_script(self, config: Dict[str, Any]) -> str:
        """Create Render startup script"""
        
        script_content = f"""#!/bin/bash
# NIS System Render Startup Script
# Generated by NIS Core Toolkit

echo "🚀 Starting NIS System on Render..."
echo "Environment: $NIS_ENV"
echo "Port: $NIS_PORT"

# Install dependencies
pip install -r requirements.txt

# Start the application
python main.py
"""
        
        script_path = self.project_root / "start.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return "start.sh"
    
    def _create_docker_run_script(self, image_name: str, config: Dict[str, Any]) -> str:
        """Create Docker run script"""
        
        script_content = f"""#!/bin/bash
# NIS System Docker Run Script
# Generated by NIS Core Toolkit

docker run -d \\
  --name nis-system \\
  -p {config['deployment']['port']}:{config['deployment']['port']} \\
  -v "$(pwd)/logs:/app/logs" \\
  -v "$(pwd)/data:/app/data" \\
  --restart unless-stopped \\
  {image_name}

echo "🚀 NIS System started!"
echo "Access at: http://localhost:{config['deployment']['port']}"
echo "Logs: docker logs -f nis-system"
echo "Stop: docker stop nis-system"
"""
        
        script_path = self.project_root / "run-docker.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return "run-docker.sh"
    
    def _display_deployment_results(self, result: Dict[str, Any], platform: str):
        """Display deployment results"""
        
        console.print(f"\n🎯 Deployment Results for {platform.upper()}", style="bold blue")
        console.print("=" * 50)
        
        if result["status"] == "success":
            console.print("✅ Deployment prepared successfully!", style="bold green")
            
            if "url" in result:
                console.print(f"🌐 URL: {result['url']}")
            
            if "instructions" in result:
                console.print("\n📋 Next Steps:", style="bold yellow")
                for i, instruction in enumerate(result["instructions"], 1):
                    console.print(f"  {i}. {instruction}")
            
            # Show generated files
            generated_files = []
            for key, value in result.items():
                if key.endswith(('_file', '_script', '_config')) and isinstance(value, str):
                    generated_files.append(value)
            
            if generated_files:
                console.print(f"\n📄 Generated Files:", style="bold cyan")
                for file in generated_files:
                    console.print(f"  • {file}")
        
        else:
            console.print("❌ Deployment failed!", style="bold red")
            if "error" in result:
                console.print(f"Error: {result['error']}", style="red")
    
    def list_platforms(self):
        """List available deployment platforms"""
        
        console.print("🚀 Available Deployment Platforms", style="bold blue")
        console.print("=" * 50)
        
        table = Table(title="Deployment Platforms")
        table.add_column("Platform", style="cyan", no_wrap=True)
        table.add_column("Description", style="magenta")
        table.add_column("Complexity", style="yellow")
        
        complexity_map = {
            "local": "Simple",
            "docker": "Medium",
            "docker-compose": "Medium",
            "render": "Easy",
            "railway": "Easy",
            "heroku": "Easy"
        }
        
        for platform, description in self.supported_platforms.items():
            complexity = complexity_map.get(platform, "Medium")
            table.add_row(platform, description, complexity)
        
        console.print(table)

def deploy_system(platform: str = "local", config_file: str = None, project_path: str = "."):
    """Main deployment function"""
    
    deployer = NISDeployer(Path(project_path))
    return deployer.deploy_system(platform, config_file)

def list_platforms():
    """List available deployment platforms"""
    
    deployer = NISDeployer()
    deployer.list_platforms()

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NIS System Deployment")
    parser.add_argument("--platform", default="local", help="Deployment platform")
    parser.add_argument("--config", help="Deployment configuration file")
    parser.add_argument("--project", default=".", help="Project directory")
    parser.add_argument("--list", action="store_true", help="List available platforms")
    
    args = parser.parse_args()
    
    if args.list:
        list_platforms()
    else:
        result = deploy_system(args.platform, args.config, args.project)
        
        if result["status"] == "success":
            console.print("\n🎉 Deployment completed successfully!", style="bold green")
        else:
            console.print("\n💥 Deployment failed!", style="bold red")
            if "error" in result:
                console.print(f"Error: {result['error']}", style="red")

if __name__ == "__main__":
    main()
