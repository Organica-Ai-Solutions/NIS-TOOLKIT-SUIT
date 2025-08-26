"""
NIS CLI Serve Command
Start development server
"""

from .base import BaseCommand
from ..utils.logger import success, error, info, step, header

class ServeCommand(BaseCommand):
    """Start development server"""
    
    @classmethod
    def register(cls, subparsers):
        parser = subparsers.add_parser('serve', help='Start development server')
        parser.add_argument('--port', '-p', type=int, default=8000, help='Port to bind')
        parser.add_argument('--hot-reload', action='store_true', help='Enable hot reload')
    
    def execute(self, args) -> int:
        header("🚀 NIS Development Server")
        project_root = self.ensure_project_root()
        
        try:
            cmd = ['python', 'main.py']
            info(f"Starting server on port {args.port}")
            self.run_command(cmd, cwd=str(project_root))
            return 0
        except Exception as e:
            error(f"Server failed: {e}")
            return 1
