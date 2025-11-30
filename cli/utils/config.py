"""
NIS CLI Configuration Management
Handles configuration loading, validation, and defaults
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class NISConfig:
    """NIS Configuration container"""
    
    # Project settings
    project_name: str = "nis-project"
    project_version: str = "1.0.0"
    project_description: str = "NIS Protocol Application"
    
    # Environment settings
    environment: str = "development"
    debug: bool = True
    
    # NIS specific settings
    nis_version: str = "4.0.0"
    enable_consciousness: bool = True
    enable_provider_router: bool = True
    enable_edge_computing: bool = False
    
    # Docker settings
    docker_image: str = "nis-toolkit"
    docker_tag: str = "latest"
    docker_registry: str = ""
    
    # Deployment settings
    deployment_target: str = "docker"
    kubernetes_namespace: str = "nis-toolkit"
    
    # Monitoring settings
    enable_monitoring: bool = True
    prometheus_port: int = 9090
    grafana_port: int = 3000
    
    # Development settings
    dev_port: int = 8000
    hot_reload: bool = True
    auto_install_deps: bool = True
    
    # Testing settings
    test_framework: str = "pytest"
    enable_coverage: bool = True
    coverage_threshold: float = 80.0
    
    # Security settings
    enable_security_scan: bool = True
    allowed_origins: list = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    
    # Custom settings (for extensibility)
    custom: Dict[str, Any] = field(default_factory=dict)

def load_config(config_path: Optional[str] = None) -> NISConfig:
    """
    Load NIS configuration from file or environment variables
    
    Priority order:
    1. Explicit config file path
    2. nis.config.yaml in current directory
    3. nis.config.json in current directory  
    4. Environment variables
    5. Default values
    """
    config = NISConfig()
    
    # Try to load from file
    if config_path:
        config_file = Path(config_path)
    else:
        # Look for config files in current directory
        yaml_config = Path("nis.config.yaml")
        json_config = Path("nis.config.json")
        
        if yaml_config.exists():
            config_file = yaml_config
        elif json_config.exists():
            config_file = json_config
        else:
            config_file = None
    
    if config_file and config_file.exists():
        try:
            config = load_config_file(config_file)
            logging.debug(f"Loaded configuration from {config_file}")
        except Exception as e:
            logging.warning(f"Failed to load config file {config_file}: {e}")
    
    # Override with environment variables
    config = apply_env_overrides(config)
    
    return config

def load_config_file(config_path: Path) -> NISConfig:
    """Load configuration from YAML or JSON file"""
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Convert dict to NISConfig
    config = NISConfig()
    
    # Apply loaded values to config
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Store unknown keys in custom dict
            config.custom[key] = value
    
    return config

def apply_env_overrides(config: NISConfig) -> NISConfig:
    """Apply environment variable overrides"""
    
    env_mappings = {
        'NIS_PROJECT_NAME': 'project_name',
        'NIS_PROJECT_VERSION': 'project_version',
        'NIS_ENVIRONMENT': 'environment',
        'NIS_DEBUG': 'debug',
        'NIS_VERSION': 'nis_version',
        'NIS_ENABLE_CONSCIOUSNESS': 'enable_consciousness',
        'NIS_ENABLE_PROVIDER_ROUTER': 'enable_provider_router',
        'NIS_ENABLE_EDGE': 'enable_edge_computing',
        'DOCKER_IMAGE': 'docker_image',
        'DOCKER_TAG': 'docker_tag',
        'DOCKER_REGISTRY': 'docker_registry',
        'DEPLOYMENT_TARGET': 'deployment_target',
        'K8S_NAMESPACE': 'kubernetes_namespace',
        'ENABLE_MONITORING': 'enable_monitoring',
        'PROMETHEUS_PORT': 'prometheus_port',
        'GRAFANA_PORT': 'grafana_port',
        'DEV_PORT': 'dev_port',
        'HOT_RELOAD': 'hot_reload',
        'AUTO_INSTALL_DEPS': 'auto_install_deps',
        'TEST_FRAMEWORK': 'test_framework',
        'ENABLE_COVERAGE': 'enable_coverage',
        'COVERAGE_THRESHOLD': 'coverage_threshold',
        'ENABLE_SECURITY_SCAN': 'enable_security_scan',
    }
    
    for env_var, config_attr in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Convert string values to appropriate types
            current_value = getattr(config, config_attr)
            
            if isinstance(current_value, bool):
                # Convert to boolean
                value = value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(current_value, int):
                # Convert to integer
                try:
                    value = int(value)
                except ValueError:
                    logging.warning(f"Invalid integer value for {env_var}: {value}")
                    continue
            elif isinstance(current_value, float):
                # Convert to float
                try:
                    value = float(value)
                except ValueError:
                    logging.warning(f"Invalid float value for {env_var}: {value}")
                    continue
            
            setattr(config, config_attr, value)
    
    return config

def save_config(config: NISConfig, output_path: str = "nis.config.yaml"):
    """Save configuration to file"""
    
    # Convert config to dict
    config_dict = {}
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        if field_name != 'custom':  # Skip custom dict for clean output
            config_dict[field_name] = value
    
    # Add custom fields
    config_dict.update(config.custom)
    
    # Write to file
    output_file = Path(output_path)
    
    with open(output_file, 'w') as f:
        if output_file.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif output_file.suffix.lower() == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_file.suffix}")
    
    logging.info(f"Configuration saved to {output_file}")

def create_default_config(output_path: str = "nis.config.yaml"):
    """Create a default configuration file"""
    config = NISConfig()
    save_config(config, output_path)
    return config

def validate_config(config: NISConfig) -> list:
    """Validate configuration and return list of issues"""
    issues = []
    
    # Validate ports
    if not (1 <= config.dev_port <= 65535):
        issues.append(f"Invalid dev_port: {config.dev_port} (must be 1-65535)")
    
    if not (1 <= config.prometheus_port <= 65535):
        issues.append(f"Invalid prometheus_port: {config.prometheus_port} (must be 1-65535)")
    
    if not (1 <= config.grafana_port <= 65535):
        issues.append(f"Invalid grafana_port: {config.grafana_port} (must be 1-65535)")
    
    # Validate thresholds
    if not (0 <= config.coverage_threshold <= 100):
        issues.append(f"Invalid coverage_threshold: {config.coverage_threshold} (must be 0-100)")
    
    # Validate deployment target
    valid_targets = ['docker', 'kubernetes', 'local', 'edge']
    if config.deployment_target not in valid_targets:
        issues.append(f"Invalid deployment_target: {config.deployment_target} (must be one of {valid_targets})")
    
    # Validate test framework
    valid_frameworks = ['pytest', 'unittest', 'nose2']
    if config.test_framework not in valid_frameworks:
        issues.append(f"Invalid test_framework: {config.test_framework} (must be one of {valid_frameworks})")
    
    return issues
