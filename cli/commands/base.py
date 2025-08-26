"""
Base Command Class for NIS CLI
Provides common functionality for all CLI commands
"""

import os
import sys
import subprocess
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, List
from pathlib import Path

try:
    from ..utils.config import NISConfig
    from ..utils.logger import success, error, info, warning, step, progress, header
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import NISConfig
    from utils.logger import success, error, info, warning, step, progress, header

class BaseCommand(ABC):
    """Base class for all NIS CLI commands"""
    
    def __init__(self, config: NISConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @classmethod
    @abstractmethod
    def register(cls, subparsers) -> None:
        """Register command with argument parser"""
        pass
    
    @abstractmethod
    def execute(self, args) -> int:
        """Execute the command. Return 0 for success, non-zero for failure"""
        pass
    
    def run_command(self, cmd: List[str], cwd: Optional[str] = None, 
                   capture_output: bool = False, check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command with logging"""
        cmd_str = ' '.join(cmd)
        self.logger.debug(f"Running command: {cmd_str}")
        
        try:
            if capture_output:
                result = subprocess.run(
                    cmd, 
                    cwd=cwd, 
                    check=check,
                    capture_output=True,
                    text=True
                )
            else:
                result = subprocess.run(cmd, cwd=cwd, check=check)
            
            if result.returncode == 0:
                self.logger.debug(f"Command succeeded: {cmd_str}")
            else:
                self.logger.error(f"Command failed with code {result.returncode}: {cmd_str}")
                
            return result
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {cmd_str}")
            if hasattr(e, 'stderr') and e.stderr:
                self.logger.error(f"Error output: {e.stderr}")
            raise
        except FileNotFoundError:
            self.logger.error(f"Command not found: {cmd[0]}")
            raise
    
    def check_prerequisites(self, tools: List[str]) -> bool:
        """Check if required tools are available"""
        missing_tools = []
        
        for tool in tools:
            try:
                subprocess.run([tool, '--version'], 
                             capture_output=True, 
                             check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(tool)
        
        if missing_tools:
            error(f"Missing required tools: {', '.join(missing_tools)}")
            info("Please install the missing tools and try again")
            return False
        
        return True
    
    def find_project_root(self, start_path: str = ".") -> Optional[Path]:
        """Find the NIS project root directory"""
        current_path = Path(start_path).resolve()
        
        # Look for NIS-specific markers
        markers = [
            "nis.config.yaml",
            "nis.config.json", 
            "VERSION",
            "nis-core-toolkit",
            "nis-agent-toolkit"
        ]
        
        while current_path != current_path.parent:
            for marker in markers:
                if (current_path / marker).exists():
                    return current_path
            current_path = current_path.parent
        
        return None
    
    def ensure_project_root(self) -> Path:
        """Ensure we're in a NIS project directory"""
        project_root = self.find_project_root()
        if not project_root:
            error("Not in a NIS project directory")
            info("Run 'nis create project <name>' to create a new project")
            sys.exit(1)
        return project_root
    
    def check_docker(self) -> bool:
        """Check if Docker is available and running"""
        try:
            result = subprocess.run(
                ['docker', 'info'], 
                capture_output=True, 
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def check_kubernetes(self) -> bool:
        """Check if kubectl is available and configured"""
        try:
            result = subprocess.run(
                ['kubectl', 'cluster-info'], 
                capture_output=True, 
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def get_docker_compose_cmd(self) -> List[str]:
        """Get the appropriate docker-compose command"""
        # Try docker compose (new) first, then docker-compose (legacy)
        for cmd in [['docker', 'compose'], ['docker-compose']]:
            try:
                subprocess.run(
                    cmd + ['version'], 
                    capture_output=True, 
                    check=True
                )
                return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        raise FileNotFoundError("Neither 'docker compose' nor 'docker-compose' found")
    
    def install_python_dependencies(self, requirements_file: str = "requirements.txt") -> bool:
        """Install Python dependencies"""
        if not Path(requirements_file).exists():
            warning(f"Requirements file not found: {requirements_file}")
            return True
        
        try:
            step(f"Installing Python dependencies from {requirements_file}")
            self.run_command([
                sys.executable, '-m', 'pip', 'install', 
                '-r', requirements_file
            ])
            success("Python dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            error(f"Failed to install dependencies from {requirements_file}")
            return False
    
    def create_directory(self, path: str, description: str = None) -> bool:
        """Create directory with logging"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            if description:
                info(f"Created {description}: {path}")
            else:
                info(f"Created directory: {path}")
            return True
        except Exception as e:
            error(f"Failed to create directory {path}: {e}")
            return False
    
    def copy_template(self, src: str, dst: str, replacements: dict = None) -> bool:
        """Copy template file with variable replacements"""
        try:
            src_path = Path(src)
            dst_path = Path(dst)
            
            if not src_path.exists():
                error(f"Template not found: {src}")
                return False
            
            # Create destination directory
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read template
            content = src_path.read_text()
            
            # Apply replacements
            if replacements:
                for key, value in replacements.items():
                    content = content.replace(f"{{{{{key}}}}}", str(value))
            
            # Write to destination
            dst_path.write_text(content)
            info(f"Created file: {dst}")
            return True
            
        except Exception as e:
            error(f"Failed to copy template {src} to {dst}: {e}")
            return False
    
    def confirm_action(self, message: str, default: bool = False) -> bool:
        """Ask user for confirmation"""
        suffix = " [Y/n]" if default else " [y/N]"
        try:
            response = input(f"❓ {message}{suffix}: ").strip().lower()
            if not response:
                return default
            return response in ('y', 'yes', 'true', '1')
        except KeyboardInterrupt:
            print("\n🛑 Operation cancelled")
            return False
    
    def show_progress(self, items: List[Any], description: str = "Processing"):
        """Show progress for a list of items"""
        total = len(items)
        for i, item in enumerate(items, 1):
            progress(f"{description} {item}", i, total)
            yield item
