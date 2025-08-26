"""
NIS CLI Deploy Command
Deployment automation for various environments
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseCommand
from ..utils.logger import success, error, info, warning, step, header, progress
from ..utils.config import NISConfig

class DeployCommand(BaseCommand):
    """Deploy NIS applications to various environments"""
    
    @classmethod
    def register(cls, subparsers):
        """Register the deploy command"""
        parser = subparsers.add_parser(
            'deploy',
            help='Deploy NIS applications',
            description='Deploy to Docker, Kubernetes, edge devices, or cloud platforms'
        )
        
        subcommands = parser.add_subparsers(
            dest='deploy_target',
            help='Deployment target',
            metavar='<target>'
        )
        
        # Docker deployment
        docker_parser = subcommands.add_parser(
            'docker',
            help='Deploy with Docker'
        )
        docker_parser.add_argument(
            '--dev',
            action='store_true',
            help='Deploy in development mode'
        )
        docker_parser.add_argument(
            '--build',
            action='store_true',
            help='Force rebuild images'
        )
        docker_parser.add_argument(
            '--scale',
            type=int,
            default=1,
            help='Number of replicas'
        )
        
        # Kubernetes deployment
        k8s_parser = subcommands.add_parser(
            'kubernetes',
            help='Deploy to Kubernetes'
        )
        k8s_parser.add_argument(
            '--namespace', '-n',
            default='nis-toolkit',
            help='Kubernetes namespace'
        )
        k8s_parser.add_argument(
            '--context', '-c',
            help='Kubernetes context'
        )
        k8s_parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Dry run without actual deployment'
        )
        
        # Edge deployment
        edge_parser = subcommands.add_parser(
            'edge',
            help='Deploy to edge devices'
        )
        edge_parser.add_argument(
            '--device',
            help='Target device IP or hostname'
        )
        edge_parser.add_argument(
            '--arch',
            choices=['amd64', 'arm64', 'armv7'],
            default='amd64',
            help='Target architecture'
        )
        
        # Local development
        local_parser = subcommands.add_parser(
            'local',
            help='Deploy locally for development'
        )
        local_parser.add_argument(
            '--port', '-p',
            type=int,
            default=8000,
            help='Port to bind to'
        )
        local_parser.add_argument(
            '--hot-reload',
            action='store_true',
            help='Enable hot reload'
        )
    
    def execute(self, args) -> int:
        """Execute the deploy command"""
        header("🚀 NIS TOOLKIT SUIT - Deployment Manager")
        
        if not args.deploy_target:
            error("Please specify deployment target (docker, kubernetes, edge, local)")
            return 1
        
        # Ensure we're in a project
        project_root = self.ensure_project_root()
        
        try:
            if args.deploy_target == 'docker':
                return self._deploy_docker(args, project_root)
            elif args.deploy_target == 'kubernetes':
                return self._deploy_kubernetes(args, project_root)
            elif args.deploy_target == 'edge':
                return self._deploy_edge(args, project_root)
            elif args.deploy_target == 'local':
                return self._deploy_local(args, project_root)
            else:
                error(f"Unknown deployment target: {args.deploy_target}")
                return 1
                
        except Exception as e:
            error(f"Deployment failed: {e}")
            return 1
    
    def _deploy_docker(self, args, project_root: Path) -> int:
        """Deploy with Docker"""
        step("Deploying with Docker...")
        
        # Check Docker availability
        if not self.check_docker():
            error("Docker is not available or not running")
            info("Please install Docker and ensure it's running")
            return 1
        
        # Get docker-compose command
        try:
            compose_cmd = self.get_docker_compose_cmd()
        except FileNotFoundError:
            error("Docker Compose not available")
            info("Please install Docker Compose")
            return 1
        
        # Choose compose file
        compose_files = ['-f', 'docker-compose.yml']
        if args.dev:
            if (project_root / 'docker-compose.dev.yml').exists():
                compose_files.extend(['-f', 'docker-compose.dev.yml'])
                info("Using development configuration")
        
        try:
            # Build if requested or if images don't exist
            if args.build or self._should_build_images():
                step("Building Docker images...")
                build_cmd = compose_cmd + compose_files + ['build']
                self.run_command(build_cmd, cwd=str(project_root))
                success("Images built successfully")
            
            # Scale services
            scale_args = []
            if args.scale > 1:
                # Get main service name from config
                service_name = self._get_main_service_name()
                scale_args = ['--scale', f'{service_name}={args.scale}']
                info(f"Scaling {service_name} to {args.scale} replicas")
            
            # Deploy
            step("Starting services...")
            up_cmd = compose_cmd + compose_files + ['up', '-d'] + scale_args
            self.run_command(up_cmd, cwd=str(project_root))
            
            # Wait for services to be ready
            self._wait_for_services(compose_cmd, compose_files, project_root)
            
            # Show status
            self._show_docker_status(compose_cmd, compose_files, project_root)
            
            success("🎉 Docker deployment completed successfully!")
            self._show_access_urls()
            
            return 0
            
        except subprocess.CalledProcessError as e:
            error(f"Docker deployment failed: {e}")
            return 1
    
    def _deploy_kubernetes(self, args, project_root: Path) -> int:
        """Deploy to Kubernetes"""
        step("Deploying to Kubernetes...")
        
        # Check kubectl availability
        if not self.check_kubernetes():
            error("kubectl is not available or cluster not accessible")
            info("Please install kubectl and configure cluster access")
            return 1
        
        # Set context if specified
        if args.context:
            try:
                self.run_command(['kubectl', 'config', 'use-context', args.context])
                info(f"Switched to context: {args.context}")
            except subprocess.CalledProcessError:
                error(f"Failed to switch to context: {args.context}")
                return 1
        
        # Find Kubernetes manifests
        k8s_dir = project_root / 'docker' / 'kubernetes'
        if not k8s_dir.exists():
            error("No Kubernetes manifests found")
            info("Expected directory: docker/kubernetes/")
            return 1
        
        try:
            # Create namespace
            step(f"Creating namespace: {args.namespace}")
            namespace_cmd = [
                'kubectl', 'create', 'namespace', args.namespace,
                '--dry-run=client', '-o', 'yaml'
            ]
            if not args.dry_run:
                namespace_cmd.remove('--dry-run=client')
            
            try:
                self.run_command(namespace_cmd + ['|', 'kubectl', 'apply', '-f', '-'])
            except subprocess.CalledProcessError:
                # Namespace might already exist
                pass
            
            # Apply manifests
            step("Applying Kubernetes manifests...")
            manifest_files = list(k8s_dir.glob('*.yaml')) + list(k8s_dir.glob('*.yml'))
            
            for manifest in manifest_files:
                apply_cmd = ['kubectl', 'apply', '-f', str(manifest), '-n', args.namespace]
                if args.dry_run:
                    apply_cmd.append('--dry-run=client')
                
                self.run_command(apply_cmd)
                info(f"Applied: {manifest.name}")
            
            if not args.dry_run:
                # Wait for deployment
                step("Waiting for deployment to be ready...")
                self.run_command([
                    'kubectl', 'wait', '--for=condition=available', 
                    'deployment', '--all', '-n', args.namespace,
                    '--timeout=300s'
                ])
                
                # Show status
                self._show_k8s_status(args.namespace)
                
                success("🎉 Kubernetes deployment completed successfully!")
            else:
                success("🎉 Kubernetes dry-run completed successfully!")
            
            return 0
            
        except subprocess.CalledProcessError as e:
            error(f"Kubernetes deployment failed: {e}")
            return 1
    
    def _deploy_edge(self, args, project_root: Path) -> int:
        """Deploy to edge devices"""
        step("Deploying to edge device...")
        
        if not args.device:
            error("Please specify target device with --device")
            return 1
        
        # Build edge-optimized image
        step("Building edge-optimized image...")
        build_cmd = [
            'docker', 'build', 
            '--target', 'edge',
            '--platform', f'linux/{args.arch}',
            '-t', f'nis-edge-{args.arch}:latest',
            '.'
        ]
        
        try:
            self.run_command(build_cmd, cwd=str(project_root))
            
            # Save image for transfer
            step("Saving image for transfer...")
            save_cmd = [
                'docker', 'save', '-o', 'nis-edge.tar',
                f'nis-edge-{args.arch}:latest'
            ]
            self.run_command(save_cmd, cwd=str(project_root))
            
            # Transfer to device
            step(f"Transferring to device: {args.device}")
            transfer_cmd = [
                'scp', 'nis-edge.tar', 
                f'{args.device}:/tmp/nis-edge.tar'
            ]
            self.run_command(transfer_cmd, cwd=str(project_root))
            
            # Deploy on device
            step("Deploying on edge device...")
            remote_cmd = [
                'ssh', args.device,
                'docker load -i /tmp/nis-edge.tar && '
                'docker stop nis-edge || true && '
                'docker rm nis-edge || true && '
                f'docker run -d --name nis-edge -p 8000:8000 nis-edge-{args.arch}:latest'
            ]
            self.run_command(remote_cmd)
            
            # Cleanup
            os.remove(project_root / 'nis-edge.tar')
            
            success(f"🎉 Edge deployment to {args.device} completed successfully!")
            info(f"Access your application at: http://{args.device}:8000")
            
            return 0
            
        except subprocess.CalledProcessError as e:
            error(f"Edge deployment failed: {e}")
            return 1
    
    def _deploy_local(self, args, project_root: Path) -> int:
        """Deploy locally for development"""
        step("Starting local development server...")
        
        # Install dependencies if needed
        if self.config.auto_install_deps:
            requirements_file = project_root / 'requirements.txt'
            if requirements_file.exists():
                if not self.install_python_dependencies(str(requirements_file)):
                    return 1
        
        # Start development server
        try:
            cmd = ['python', 'main.py']
            env = os.environ.copy()
            env.update({
                'NIS_ENVIRONMENT': 'development',
                'NIS_DEBUG': 'true',
                'DEV_PORT': str(args.port),
                'HOT_RELOAD': str(args.hot_reload).lower()
            })
            
            info(f"Starting development server on port {args.port}")
            if args.hot_reload:
                info("Hot reload enabled")
            
            success("🎉 Local development server started!")
            info(f"Access your application at: http://localhost:{args.port}")
            info("Press Ctrl+C to stop")
            
            # Run the server
            process = subprocess.run(cmd, cwd=str(project_root), env=env)
            return process.returncode
            
        except KeyboardInterrupt:
            info("\n🛑 Development server stopped")
            return 0
        except subprocess.CalledProcessError as e:
            error(f"Failed to start development server: {e}")
            return 1
    
    def _should_build_images(self) -> bool:
        """Check if images need to be built"""
        # Simple check - always build for now
        # Could be enhanced to check for image existence and freshness
        return True
    
    def _get_main_service_name(self) -> str:
        """Get the main service name from docker-compose"""
        # Could parse docker-compose.yml to find main service
        # For now, assume standard naming
        return self.config.project_name.lower().replace('-', '_')
    
    def _wait_for_services(self, compose_cmd: List[str], compose_files: List[str], project_root: Path):
        """Wait for Docker services to be ready"""
        step("Waiting for services to be ready...")
        
        # Wait a bit for services to start
        time.sleep(5)
        
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                # Check if services are healthy
                ps_cmd = compose_cmd + compose_files + ['ps']
                result = self.run_command(ps_cmd, cwd=str(project_root), capture_output=True)
                
                # Simple check - if command succeeds, services are probably ready
                break
            except:
                if attempt < max_attempts - 1:
                    time.sleep(2)
                else:
                    warning("Services may not be fully ready yet")
    
    def _show_docker_status(self, compose_cmd: List[str], compose_files: List[str], project_root: Path):
        """Show Docker deployment status"""
        step("Deployment status:")
        
        try:
            ps_cmd = compose_cmd + compose_files + ['ps']
            self.run_command(ps_cmd, cwd=str(project_root))
        except:
            warning("Could not get service status")
    
    def _show_k8s_status(self, namespace: str):
        """Show Kubernetes deployment status"""
        step("Deployment status:")
        
        try:
            # Show pods
            self.run_command(['kubectl', 'get', 'pods', '-n', namespace])
            
            # Show services
            self.run_command(['kubectl', 'get', 'services', '-n', namespace])
            
            # Show ingress if any
            try:
                self.run_command(['kubectl', 'get', 'ingress', '-n', namespace])
            except:
                pass  # Ingress might not exist
        except:
            warning("Could not get deployment status")
    
    def _show_access_urls(self):
        """Show access URLs for the deployed application"""
        header("🌐 Access URLs")
        info(f"🎯 Core Service:      http://localhost:8000")
        info(f"🤖 Agent Service:     http://localhost:8001")
        info(f"📱 Edge Service:      http://localhost:8002")
        info(f"📊 Prometheus:        http://localhost:9090")
        info(f"📈 Grafana:           http://localhost:3000")
        info(f"📋 Jupyter Lab:       http://localhost:8888")
        info("")
        info("📋 Useful Commands:")
        info("  View logs:          nis monitor logs")
        info("  Scale services:     nis deploy docker --scale 3")
        info("  Stop services:      docker-compose down")
        info("  Update deployment:  nis deploy docker --build")
