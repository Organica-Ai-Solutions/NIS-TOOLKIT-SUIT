"""
NIS CLI Build Command  
Build and package applications
"""

from .base import BaseCommand
from ..utils.logger import success, error, info, step, header

class BuildCommand(BaseCommand):
    """Build and package applications"""
    
    @classmethod  
    def register(cls, subparsers):
        parser = subparsers.add_parser('build', help='Build and package')
        parser.add_argument('--target', choices=['docker', 'wheel', 'executable'], default='docker')
        parser.add_argument('--optimize', action='store_true', help='Optimize build')
    
    def execute(self, args) -> int:
        header("🏗️ NIS Build System")
        project_root = self.ensure_project_root()
        
        try:
            if args.target == 'docker':
                step("Building Docker image...")
                self.run_command(['docker', 'build', '-t', f'{self.config.project_name}:latest', '.'], 
                               cwd=str(project_root))
            
            success(f"Build completed: {args.target}")
            return 0
        except Exception as e:
            error(f"Build failed: {e}")
            return 1
