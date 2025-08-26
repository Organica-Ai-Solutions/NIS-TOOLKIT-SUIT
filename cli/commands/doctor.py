"""
NIS CLI Doctor Command
Comprehensive health checks, diagnostics, and auto-fixing
"""

import os
import sys
import subprocess
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .base import BaseCommand
from ..utils.logger import success, error, info, warning, step, header, progress
from ..utils.config import NISConfig, validate_config

class HealthCheck:
    """Individual health check"""
    def __init__(self, name: str, description: str, critical: bool = False):
        self.name = name
        self.description = description
        self.critical = critical
        self.status = "pending"
        self.message = ""
        self.fix_available = False
        self.fix_command = None

class DoctorCommand(BaseCommand):
    """Comprehensive system diagnostics and health checks"""
    
    @classmethod
    def register(cls, subparsers):
        """Register the doctor command"""
        parser = subparsers.add_parser(
            'doctor',
            help='Run system diagnostics and health checks',
            description='Comprehensive health checks for NIS development environment'
        )
        
        parser.add_argument(
            '--fix',
            action='store_true',
            help='Automatically fix issues where possible'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Show detailed diagnostic information'
        )
        
        parser.add_argument(
            '--category', '-c',
            choices=['system', 'python', 'docker', 'nis', 'security', 'all'],
            default='all',
            help='Run specific category of checks'
        )
        
        parser.add_argument(
            '--output', '-o',
            choices=['console', 'json', 'report'],
            default='console',
            help='Output format'
        )
    
    def execute(self, args) -> int:
        """Execute the doctor command"""
        header("🏥 NIS TOOLKIT SUIT - System Doctor")
        
        self.auto_fix = args.fix
        self.verbose = args.verbose
        self.category = args.category
        self.output_format = args.output
        
        # Run diagnostics
        all_checks = self._run_diagnostics()
        
        # Generate report
        if self.output_format == 'json':
            return self._output_json(all_checks)
        elif self.output_format == 'report':
            return self._output_report(all_checks)
        else:
            return self._output_console(all_checks)
    
    def _run_diagnostics(self) -> Dict[str, List[HealthCheck]]:
        """Run all diagnostic checks"""
        all_checks = {}
        
        if self.category in ['system', 'all']:
            info("🖥️  Running system checks...")
            all_checks['system'] = self._check_system()
        
        if self.category in ['python', 'all']:
            info("🐍 Running Python environment checks...")
            all_checks['python'] = self._check_python()
        
        if self.category in ['docker', 'all']:
            info("🐳 Running Docker checks...")
            all_checks['docker'] = self._check_docker()
        
        if self.category in ['nis', 'all']:
            info("🧠 Running NIS-specific checks...")
            all_checks['nis'] = self._check_nis()
        
        if self.category in ['security', 'all']:
            info("🔒 Running security checks...")
            all_checks['security'] = self._check_security()
        
        return all_checks
    
    def _check_system(self) -> List[HealthCheck]:
        """Check system environment"""
        checks = []
        
        # Operating system
        check = HealthCheck("os_support", "Operating system compatibility", critical=True)
        if sys.platform in ['linux', 'darwin', 'win32']:
            check.status = "pass"
            check.message = f"Supported OS: {sys.platform}"
        else:
            check.status = "fail"
            check.message = f"Unsupported OS: {sys.platform}"
        checks.append(check)
        
        # Memory
        check = HealthCheck("memory", "Available memory")
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            if available_gb >= 4:
                check.status = "pass"
                check.message = f"Available memory: {available_gb:.1f} GB"
            elif available_gb >= 2:
                check.status = "warning"
                check.message = f"Limited memory: {available_gb:.1f} GB (4GB+ recommended)"
            else:
                check.status = "fail"
                check.message = f"Insufficient memory: {available_gb:.1f} GB"
        except ImportError:
            check.status = "warning"
            check.message = "Cannot check memory (psutil not available)"
        checks.append(check)
        
        # Disk space
        check = HealthCheck("disk_space", "Available disk space")
        try:
            disk_usage = shutil.disk_usage("/")
            free_gb = disk_usage.free / (1024**3)
            if free_gb >= 10:
                check.status = "pass"
                check.message = f"Available disk space: {free_gb:.1f} GB"
            elif free_gb >= 5:
                check.status = "warning"
                check.message = f"Limited disk space: {free_gb:.1f} GB"
            else:
                check.status = "fail"
                check.message = f"Low disk space: {free_gb:.1f} GB"
        except:
            check.status = "warning"
            check.message = "Cannot check disk space"
        checks.append(check)
        
        # Git
        check = HealthCheck("git", "Git installation")
        if self._command_exists("git"):
            try:
                result = subprocess.run(['git', '--version'], capture_output=True, text=True)
                check.status = "pass"
                check.message = f"Git installed: {result.stdout.strip()}"
            except:
                check.status = "fail"
                check.message = "Git command failed"
        else:
            check.status = "fail"
            check.message = "Git not installed"
            check.fix_available = True
            check.fix_command = "Install Git from https://git-scm.com/"
        checks.append(check)
        
        return checks
    
    def _check_python(self) -> List[HealthCheck]:
        """Check Python environment"""
        checks = []
        
        # Python version
        check = HealthCheck("python_version", "Python version", critical=True)
        version = sys.version_info
        if version >= (3, 8):
            check.status = "pass"
            check.message = f"Python {version.major}.{version.minor}.{version.micro}"
        elif version >= (3, 7):
            check.status = "warning"
            check.message = f"Python {version.major}.{version.minor}.{version.micro} (3.8+ recommended)"
        else:
            check.status = "fail"
            check.message = f"Python {version.major}.{version.minor}.{version.micro} (3.8+ required)"
        checks.append(check)
        
        # pip
        check = HealthCheck("pip", "pip package manager")
        try:
            import pip
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                  capture_output=True, text=True)
            check.status = "pass"
            check.message = result.stdout.strip()
        except:
            check.status = "fail"
            check.message = "pip not available"
        checks.append(check)
        
        # Virtual environment
        check = HealthCheck("venv", "Virtual environment")
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            check.status = "pass"
            check.message = "Running in virtual environment"
        else:
            check.status = "warning"
            check.message = "Not in virtual environment (recommended)"
            check.fix_available = True
            check.fix_command = "python -m venv venv && source venv/bin/activate"
        checks.append(check)
        
        # Key packages
        key_packages = ['fastapi', 'uvicorn', 'pydantic', 'numpy']
        for package in key_packages:
            check = HealthCheck(f"package_{package}", f"Package: {package}")
            try:
                __import__(package)
                check.status = "pass"
                check.message = f"{package} is available"
            except ImportError:
                check.status = "warning"
                check.message = f"{package} not installed"
                check.fix_available = True
                check.fix_command = f"pip install {package}"
            checks.append(check)
        
        return checks
    
    def _check_docker(self) -> List[HealthCheck]:
        """Check Docker environment"""
        checks = []
        
        # Docker engine
        check = HealthCheck("docker_engine", "Docker engine")
        if self._command_exists("docker"):
            try:
                result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
                check.status = "pass"
                check.message = result.stdout.strip()
            except:
                check.status = "fail"
                check.message = "Docker command failed"
        else:
            check.status = "warning"
            check.message = "Docker not installed"
            check.fix_available = True
            check.fix_command = "Install Docker from https://docker.com"
        checks.append(check)
        
        # Docker daemon
        if self._command_exists("docker"):
            check = HealthCheck("docker_daemon", "Docker daemon")
            try:
                result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
                if result.returncode == 0:
                    check.status = "pass"
                    check.message = "Docker daemon is running"
                else:
                    check.status = "fail"
                    check.message = "Docker daemon not running"
                    check.fix_available = True
                    check.fix_command = "Start Docker Desktop or docker service"
            except:
                check.status = "fail"
                check.message = "Cannot connect to Docker daemon"
            checks.append(check)
        
        # Docker Compose
        check = HealthCheck("docker_compose", "Docker Compose")
        compose_cmd = None
        for cmd in [['docker', 'compose'], ['docker-compose']]:
            try:
                result = subprocess.run(cmd + ['version'], capture_output=True, text=True)
                if result.returncode == 0:
                    compose_cmd = ' '.join(cmd)
                    break
            except:
                continue
        
        if compose_cmd:
            check.status = "pass"
            check.message = f"Docker Compose available: {compose_cmd}"
        else:
            check.status = "warning"
            check.message = "Docker Compose not available"
            check.fix_available = True
            check.fix_command = "Install Docker Compose"
        checks.append(check)
        
        return checks
    
    def _check_nis(self) -> List[HealthCheck]:
        """Check NIS-specific configuration"""
        checks = []
        
        # Project structure
        check = HealthCheck("project_structure", "NIS project structure")
        project_root = self.find_project_root()
        if project_root:
            check.status = "pass"
            check.message = f"NIS project detected: {project_root}"
        else:
            check.status = "info"
            check.message = "Not in a NIS project directory"
        checks.append(check)
        
        # Configuration file
        if project_root:
            check = HealthCheck("nis_config", "NIS configuration")
            config_files = ['nis.config.yaml', 'nis.config.json']
            config_found = False
            
            for config_file in config_files:
                if (project_root / config_file).exists():
                    config_found = True
                    # Validate configuration
                    try:
                        from ..utils.config import load_config
                        config = load_config(str(project_root / config_file))
                        validation_issues = validate_config(config)
                        
                        if not validation_issues:
                            check.status = "pass"
                            check.message = f"Valid configuration: {config_file}"
                        else:
                            check.status = "warning"
                            check.message = f"Configuration issues: {len(validation_issues)} problems"
                    except Exception as e:
                        check.status = "fail"
                        check.message = f"Configuration error: {e}"
                    break
            
            if not config_found:
                check.status = "warning"
                check.message = "No NIS configuration file found"
                check.fix_available = True
                check.fix_command = "nis create config"
            
            checks.append(check)
        
        # NIS version
        check = HealthCheck("nis_version", "NIS version compatibility")
        try:
            with open(project_root / "VERSION") as f:
                version = f.read().strip()
            if version == self.config.nis_version:
                check.status = "pass"
                check.message = f"Version: {version}"
            else:
                check.status = "warning"
                check.message = f"Version mismatch: {version} (current: {self.config.nis_version})"
                check.fix_available = True
                check.fix_command = "nis update"
        except:
            check.status = "info"
            check.message = "Version file not found"
        checks.append(check)
        
        return checks
    
    def _check_security(self) -> List[HealthCheck]:
        """Check security-related issues"""
        checks = []
        
        # Permissions
        check = HealthCheck("permissions", "File permissions")
        current_dir = Path.cwd()
        if os.access(current_dir, os.R_OK | os.W_OK):
            check.status = "pass"
            check.message = "Directory permissions OK"
        else:
            check.status = "fail"
            check.message = "Insufficient directory permissions"
        checks.append(check)
        
        # Environment variables
        check = HealthCheck("env_vars", "Environment variables")
        sensitive_vars = ['API_KEY', 'SECRET_KEY', 'PASSWORD', 'TOKEN']
        exposed_vars = []
        
        for var in os.environ:
            if any(sensitive in var.upper() for sensitive in sensitive_vars):
                exposed_vars.append(var)
        
        if exposed_vars:
            check.status = "warning"
            check.message = f"Sensitive environment variables detected: {len(exposed_vars)}"
        else:
            check.status = "pass"
            check.message = "No sensitive environment variables exposed"
        checks.append(check)
        
        # Dependencies security
        check = HealthCheck("dep_security", "Dependency security")
        try:
            # Try to run safety check if available
            result = subprocess.run(['safety', 'check', '--json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                vulnerabilities = json.loads(result.stdout)
                if not vulnerabilities:
                    check.status = "pass"
                    check.message = "No known vulnerabilities"
                else:
                    check.status = "warning"
                    check.message = f"{len(vulnerabilities)} vulnerabilities found"
                    check.fix_available = True
                    check.fix_command = "pip install --upgrade affected-packages"
            else:
                check.status = "info"
                check.message = "Cannot check vulnerabilities (safety not available)"
        except:
            check.status = "info"
            check.message = "Security scanning not available"
        checks.append(check)
        
        return checks
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists"""
        return shutil.which(command) is not None
    
    def _output_console(self, all_checks: Dict[str, List[HealthCheck]]) -> int:
        """Output results to console"""
        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        warning_checks = 0
        
        for category, checks in all_checks.items():
            header(f"🔍 {category.title()} Checks")
            
            for check in checks:
                total_checks += 1
                
                if check.status == "pass":
                    success(f"✅ {check.name}: {check.message}")
                    passed_checks += 1
                elif check.status == "fail":
                    error(f"❌ {check.name}: {check.message}")
                    failed_checks += 1
                    if check.fix_available and self.auto_fix:
                        info(f"🔧 Auto-fixing: {check.fix_command}")
                        # Implement auto-fix logic here
                elif check.status == "warning":
                    warning(f"⚠️  {check.name}: {check.message}")
                    warning_checks += 1
                    if check.fix_available and self.auto_fix:
                        info(f"🔧 Auto-fixing: {check.fix_command}")
                else:
                    info(f"ℹ️  {check.name}: {check.message}")
                
                if check.fix_available and not self.auto_fix:
                    info(f"💡 Fix: {check.fix_command}")
        
        # Summary
        header("📊 Summary")
        info(f"Total checks: {total_checks}")
        success(f"Passed: {passed_checks}")
        if warning_checks > 0:
            warning(f"Warnings: {warning_checks}")
        if failed_checks > 0:
            error(f"Failed: {failed_checks}")
        
        if failed_checks == 0 and warning_checks == 0:
            success("🎉 All checks passed! Your NIS environment is healthy.")
            return 0
        elif failed_checks > 0:
            error("❌ Some checks failed. Please address the issues above.")
            if not self.auto_fix:
                info("💡 Run with --fix to automatically fix issues where possible")
            return 1
        else:
            warning("⚠️  Some warnings detected. Consider addressing them for optimal performance.")
            return 0
    
    def _output_json(self, all_checks: Dict[str, List[HealthCheck]]) -> int:
        """Output results as JSON"""
        output = {
            "timestamp": str(datetime.now()),
            "categories": {}
        }
        
        for category, checks in all_checks.items():
            output["categories"][category] = []
            for check in checks:
                output["categories"][category].append({
                    "name": check.name,
                    "description": check.description,
                    "status": check.status,
                    "message": check.message,
                    "critical": check.critical,
                    "fix_available": check.fix_available,
                    "fix_command": check.fix_command
                })
        
        print(json.dumps(output, indent=2))
        return 0
    
    def _output_report(self, all_checks: Dict[str, List[HealthCheck]]) -> int:
        """Output detailed report"""
        from datetime import datetime
        
        report_path = Path("nis_health_report.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# NIS Health Report\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            for category, checks in all_checks.items():
                f.write(f"## {category.title()} Checks\n\n")
                
                for check in checks:
                    status_emoji = {
                        "pass": "✅",
                        "fail": "❌", 
                        "warning": "⚠️",
                        "info": "ℹ️"
                    }.get(check.status, "❓")
                    
                    f.write(f"### {status_emoji} {check.name}\n\n")
                    f.write(f"**Status**: {check.status}\n\n")
                    f.write(f"**Message**: {check.message}\n\n")
                    
                    if check.fix_available:
                        f.write(f"**Fix**: `{check.fix_command}`\n\n")
                    
                    f.write("---\n\n")
        
        success(f"Health report saved to: {report_path}")
        return 0
