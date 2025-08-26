#!/usr/bin/env python3
"""
NIS TOOLKIT SUIT v3.2.1 - Master CLI
Universal command-line interface for NIS development, deployment, and management.

Usage:
    nis <command> [options]

Commands:
    create      Create new NIS project or component
    deploy      Deploy NIS applications to various environments  
    test        Run comprehensive testing and validation
    monitor     Real-time monitoring and observability
    optimize    Performance optimization and tuning
    doctor      Health check and diagnostics
    update      Update NIS components and dependencies
    migrate     Migrate between NIS versions
    serve       Start development server
    build       Build and package applications
    
For detailed help on any command:
    nis <command> --help
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the CLI directory to Python path
CLI_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(CLI_DIR))

try:
    # Try relative imports first (when run as module)
    try:
        from commands.create import CreateCommand
        from commands.deploy import DeployCommand  
        from commands.test import TestCommand
        from commands.monitor import MonitorCommand
        from commands.optimize import OptimizeCommand
        from commands.doctor import DoctorCommand
        from commands.update import UpdateCommand
        from commands.migrate import MigrateCommand
        from commands.serve import ServeCommand
        from commands.build import BuildCommand
        from utils.logger import setup_logging
        from utils.config import load_config
    except ImportError:
        # Try absolute imports (when run directly)
        sys.path.insert(0, str(CLI_DIR))
        from commands.create import CreateCommand
        from commands.deploy import DeployCommand  
        from commands.test import TestCommand
        from commands.monitor import MonitorCommand
        from commands.optimize import OptimizeCommand
        from commands.doctor import DoctorCommand
        from commands.update import UpdateCommand
        from commands.migrate import MigrateCommand
        from commands.serve import ServeCommand
        from commands.build import BuildCommand
        from utils.logger import setup_logging
        from utils.config import load_config
except ImportError as e:
    print(f"❌ Error importing CLI components: {e}")
    print("Make sure you're running from the NIS TOOLKIT SUIT directory")
    sys.exit(1)

def create_parser():
    """Create the main CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog='nis',
        description='NIS TOOLKIT SUIT v3.2.1 - Universal AI Development CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    nis create project my-ai-app                 # Create new project
    nis deploy docker --environment production   # Deploy with Docker
    nis test --coverage --benchmark              # Run tests with coverage
    nis monitor --dashboard                      # Start monitoring dashboard
    nis doctor --fix                             # Run diagnostics and auto-fix
    nis serve --hot-reload --port 8000           # Development server
    
For more information, visit: https://github.com/nis-protocol/toolkit
"""
    )
    
    parser.add_argument(
        '--version', 
        action='version',
        version='NIS TOOLKIT SUIT v3.2.1'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (use -v, -vv, or -vvv)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='<command>'
    )
    
    # Register all commands
    CreateCommand.register(subparsers)
    DeployCommand.register(subparsers)
    TestCommand.register(subparsers)
    MonitorCommand.register(subparsers)
    OptimizeCommand.register(subparsers)
    DoctorCommand.register(subparsers)
    UpdateCommand.register(subparsers)
    MigrateCommand.register(subparsers)
    ServeCommand.register(subparsers)
    BuildCommand.register(subparsers)
    
    return parser

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging based on verbosity
    log_level = max(logging.WARNING - (args.verbose * 10), logging.DEBUG)
    setup_logging(log_level, not args.no_color)
    
    # Load configuration
    config = load_config(args.config)
    
    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    try:
        command_map = {
            'create': CreateCommand,
            'deploy': DeployCommand,
            'test': TestCommand,
            'monitor': MonitorCommand,
            'optimize': OptimizeCommand,
            'doctor': DoctorCommand,
            'update': UpdateCommand,
            'migrate': MigrateCommand,
            'serve': ServeCommand,
            'build': BuildCommand,
        }
        
        command_class = command_map.get(args.command)
        if not command_class:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
        command = command_class(config)
        return command.execute(args)
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        return 130
    except Exception as e:
        logging.error(f"Command failed: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
