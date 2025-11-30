"""
Flutter Desktop Client Configuration for NIS Protocol v4.0

This module provides configuration and utilities for integrating with
the NIS Protocol Flutter desktop application.

Features:
- Flutter client connection management
- Agentic chat configuration
- Real-time telemetry streaming
- Robotics control panel integration
- BitNet status monitoring
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FlutterClientConfig:
    """Configuration for Flutter desktop client connection."""
    
    # Connection settings
    api_base_url: str = "http://localhost:8000"
    websocket_url: str = "ws://localhost:8000/ws/agentic"
    
    # Authentication
    auth_mode: str = "jwt"  # "jwt", "api_key", or "dev_bypass"
    jwt_token: Optional[str] = None
    api_key: Optional[str] = None
    dev_bypass_enabled: bool = False
    
    # Feature toggles
    enable_agentic_chat: bool = True
    enable_telemetry: bool = True
    enable_robotics_panel: bool = True
    enable_bitnet_monitor: bool = True
    enable_consciousness_viewer: bool = True
    
    # UI preferences
    theme: str = "dark"  # "dark" or "light"
    language: str = "en"
    
    # Performance settings
    telemetry_refresh_rate_ms: int = 1000
    chat_history_limit: int = 100
    max_concurrent_requests: int = 5
    request_timeout_seconds: int = 30
    
    # Robotics settings
    robotics_control_mode: str = "safe"  # "safe", "standard", "advanced"
    enable_trajectory_preview: bool = True
    enable_collision_detection: bool = True
    
    # Consciousness pipeline settings
    default_consciousness_phases: List[str] = field(default_factory=lambda: [
        "genesis", "plan", "collective", "multipath", "ethics",
        "embodiment", "evolution", "reflection", "marketplace", "debug"
    ])
    consciousness_visualization: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlutterClientConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, json_str: str) -> "FlutterClientConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def from_file(cls, path: str) -> "FlutterClientConfig":
        """Load config from file."""
        with open(path, 'r') as f:
            return cls.from_json(f.read())
    
    def save_to_file(self, path: str) -> None:
        """Save config to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())


@dataclass
class AgenticChatConfig:
    """Configuration for agentic chat functionality."""
    
    # Chat behavior
    enable_streaming: bool = True
    enable_tool_calls: bool = True
    enable_consciousness_integration: bool = True
    
    # Agent selection
    default_agent: str = "consciousness"
    available_agents: List[str] = field(default_factory=lambda: [
        "consciousness", "robotics", "physics", "research", "vision", "voice"
    ])
    
    # Message handling
    max_message_length: int = 10000
    enable_markdown: bool = True
    enable_code_highlighting: bool = True
    
    # Context management
    context_window_size: int = 10
    enable_memory: bool = True
    memory_type: str = "session"  # "session", "persistent", "hybrid"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry dashboard."""
    
    # Metrics to display
    show_cpu_usage: bool = True
    show_memory_usage: bool = True
    show_request_rate: bool = True
    show_response_time: bool = True
    show_agent_status: bool = True
    show_consciousness_state: bool = True
    
    # Refresh settings
    auto_refresh: bool = True
    refresh_interval_ms: int = 1000
    
    # History
    history_duration_seconds: int = 300
    history_resolution_ms: int = 1000
    
    # Alerts
    enable_alerts: bool = True
    cpu_alert_threshold: float = 80.0
    memory_alert_threshold: float = 85.0
    response_time_alert_ms: int = 5000
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoboticsControlConfig:
    """Configuration for robotics control panel."""
    
    # Control modes
    control_mode: str = "safe"  # "safe", "standard", "advanced"
    enable_manual_control: bool = True
    enable_trajectory_planning: bool = True
    enable_real_time_feedback: bool = True
    
    # Safety settings
    enable_collision_detection: bool = True
    enable_joint_limits: bool = True
    enable_velocity_limits: bool = True
    emergency_stop_enabled: bool = True
    
    # Visualization
    show_3d_model: bool = True
    show_trajectory_preview: bool = True
    show_workspace_bounds: bool = True
    
    # Robot configuration
    robot_type: str = "generic_6dof"
    dh_parameters: Optional[Dict[str, Any]] = None
    joint_limits: Optional[List[Dict[str, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FlutterClientManager:
    """Manager for Flutter desktop client integration."""
    
    def __init__(self, config: Optional[FlutterClientConfig] = None):
        """Initialize the Flutter client manager.
        
        Args:
            config: Flutter client configuration
        """
        self.config = config or FlutterClientConfig()
        self.chat_config = AgenticChatConfig()
        self.telemetry_config = TelemetryConfig()
        self.robotics_config = RoboticsControlConfig()
        
        self._connected = False
        self._session_id: Optional[str] = None
        
        logger.info("Flutter client manager initialized")
    
    def generate_client_config(self) -> Dict[str, Any]:
        """Generate complete configuration for Flutter client.
        
        Returns:
            Complete configuration dictionary
        """
        return {
            "version": "4.0.0",
            "client": self.config.to_dict(),
            "chat": self.chat_config.to_dict(),
            "telemetry": self.telemetry_config.to_dict(),
            "robotics": self.robotics_config.to_dict(),
            "endpoints": self._generate_endpoint_config(),
            "features": self._generate_feature_flags()
        }
    
    def _generate_endpoint_config(self) -> Dict[str, str]:
        """Generate API endpoint configuration."""
        base = self.config.api_base_url
        return {
            # Core endpoints
            "health": f"{base}/health",
            "docs": f"{base}/docs",
            "console": f"{base}/console",
            "metrics": f"{base}/metrics",
            "openapi": f"{base}/openapi.json",
            
            # Consciousness - Genesis Phase
            "consciousness_genesis": f"{base}/v4/consciousness/genesis",
            "consciousness_genesis_history": f"{base}/v4/consciousness/genesis/history",
            
            # Consciousness - Plan Phase
            "consciousness_plan": f"{base}/v4/consciousness/plan",
            "consciousness_plan_status": f"{base}/v4/consciousness/plan/status",
            
            # Consciousness - Collective Phase
            "consciousness_collective_decide": f"{base}/v4/consciousness/collective/decide",
            "consciousness_collective_register": f"{base}/v4/consciousness/collective/register",
            "consciousness_collective_status": f"{base}/v4/consciousness/collective/status",
            "consciousness_collective_sync": f"{base}/v4/consciousness/collective/sync",
            
            # Consciousness - Multipath Phase
            "consciousness_multipath_start": f"{base}/v4/consciousness/multipath/start",
            "consciousness_multipath_collapse": f"{base}/v4/consciousness/multipath/collapse",
            "consciousness_multipath_state": f"{base}/v4/consciousness/multipath/state",
            
            # Consciousness - Ethics Phase
            "consciousness_ethics": f"{base}/v4/consciousness/ethics/evaluate",
            
            # Consciousness - Embodiment Phase
            "consciousness_embodiment_status": f"{base}/v4/consciousness/embodiment/status",
            "consciousness_embodiment_action": f"{base}/v4/consciousness/embodiment/action/execute",
            "consciousness_embodiment_motion": f"{base}/v4/consciousness/embodiment/motion/check",
            "consciousness_embodiment_state": f"{base}/v4/consciousness/embodiment/state/update",
            "consciousness_embodiment_vision": f"{base}/v4/consciousness/embodiment/vision/detect",
            
            # Consciousness - Evolution Phase
            "consciousness_evolve": f"{base}/v4/consciousness/evolve",
            "consciousness_evolution_history": f"{base}/v4/consciousness/evolution/history",
            "consciousness_meta_evolve": f"{base}/v4/consciousness/meta-evolve",
            
            # Consciousness - Marketplace Phase
            "consciousness_marketplace_list": f"{base}/v4/consciousness/marketplace/list",
            "consciousness_marketplace_publish": f"{base}/v4/consciousness/marketplace/publish",
            
            # Consciousness - Debug Phase
            "consciousness_debug_explain": f"{base}/v4/consciousness/debug/explain",
            "consciousness_debug_record": f"{base}/v4/consciousness/debug/record",
            
            # Consciousness - Performance
            "consciousness_performance": f"{base}/v4/consciousness/performance",
            
            # Robotics endpoints
            "robotics_fk": f"{base}/robotics/forward_kinematics",
            "robotics_ik": f"{base}/robotics/inverse_kinematics",
            "robotics_trajectory": f"{base}/robotics/plan_trajectory",
            "robotics_capabilities": f"{base}/robotics/capabilities",
            "robotics_telemetry": f"{base}/robotics/telemetry",
            
            # Physics endpoints
            "physics_validate": f"{base}/physics/validate",
            "physics_heat": f"{base}/physics/solve/heat-equation",
            "physics_wave": f"{base}/physics/solve/wave-equation",
            "physics_constants": f"{base}/physics/constants",
            
            # Vision endpoints
            "vision_analyze": f"{base}/vision/analyze",
            "vision_generate": f"{base}/vision/generate",
            "vision_detect": f"{base}/vision/detect",
            "vision_ocr": f"{base}/vision/ocr",
            
            # Voice endpoints
            "voice_tts": f"{base}/voice/tts",
            "voice_stt": f"{base}/voice/stt",
            "voice_stream": f"{base}/voice/stream",
            "voice_voices": f"{base}/voice/voices",
            
            # Research endpoints
            "research_search": f"{base}/research/search",
            "research_arxiv": f"{base}/research/arxiv",
            
            # BitNet endpoints
            "bitnet_status": f"{base}/bitnet/status",
            "bitnet_generate": f"{base}/bitnet/generate",
            "bitnet_models": f"{base}/bitnet/models",
            
            # Auth endpoints
            "auth_login": f"{base}/auth/login",
            "auth_logout": f"{base}/auth/logout",
            "auth_verify": f"{base}/auth/verify",
            "auth_signup": f"{base}/auth/signup",
            "auth_refresh": f"{base}/auth/refresh",
            
            # User endpoints
            "users_profile": f"{base}/users/profile",
            "users_api_keys": f"{base}/users/api-keys",
            
            # Agents endpoints
            "agents_list": f"{base}/agents",
            "agents_create": f"{base}/agents/create",
            "agents_status": f"{base}/agents/status",
            
            # WebSocket endpoints
            "ws_agentic": self.config.websocket_url,
            "ws_robotics": f"{self.config.websocket_url.replace('/agentic', '/robotics/control')}",
            "ws_telemetry": f"{self.config.websocket_url.replace('/agentic', '/telemetry')}",
            "ws_voice": f"{self.config.websocket_url.replace('/agentic', '/voice/stream')}",
        }
    
    def _generate_feature_flags(self) -> Dict[str, bool]:
        """Generate feature flags for client."""
        return {
            "agentic_chat": self.config.enable_agentic_chat,
            "telemetry_dashboard": self.config.enable_telemetry,
            "robotics_control": self.config.enable_robotics_panel,
            "bitnet_monitor": self.config.enable_bitnet_monitor,
            "consciousness_viewer": self.config.enable_consciousness_viewer,
            "streaming_responses": self.chat_config.enable_streaming,
            "tool_calls": self.chat_config.enable_tool_calls,
            "3d_visualization": self.robotics_config.show_3d_model,
            "trajectory_preview": self.robotics_config.show_trajectory_preview,
            "collision_detection": self.robotics_config.enable_collision_detection,
            "alerts": self.telemetry_config.enable_alerts,
        }
    
    def export_config_for_flutter(self, output_path: str) -> str:
        """Export configuration file for Flutter client.
        
        Args:
            output_path: Path to save the configuration
            
        Returns:
            Path to the saved configuration file
        """
        config = self.generate_client_config()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Flutter client config exported to {output_path}")
        return output_path
    
    def generate_dart_config_class(self) -> str:
        """Generate Dart configuration class for Flutter.
        
        Returns:
            Dart code string for configuration class
        """
        config = self.generate_client_config()
        
        dart_code = '''
// Auto-generated NIS Protocol v4.0 Configuration
// Generated by NIS-TOOLKIT-SUIT Flutter Client Manager

class NISProtocolConfig {
  // API Configuration
  static const String apiBaseUrl = '${api_base_url}';
  static const String websocketUrl = '${websocket_url}';
  
  // Authentication
  static const String authMode = '${auth_mode}';
  static const bool devBypassEnabled = ${dev_bypass};
  
  // Feature Flags
  static const bool enableAgenticChat = ${enable_chat};
  static const bool enableTelemetry = ${enable_telemetry};
  static const bool enableRoboticsPanel = ${enable_robotics};
  static const bool enableBitnetMonitor = ${enable_bitnet};
  static const bool enableConsciousnessViewer = ${enable_consciousness};
  
  // Performance Settings
  static const int telemetryRefreshRateMs = ${telemetry_rate};
  static const int chatHistoryLimit = ${chat_limit};
  static const int requestTimeoutSeconds = ${timeout};
  
  // Consciousness Phases
  static const List<String> consciousnessPhases = [
    'genesis', 'plan', 'collective', 'multipath', 'ethics',
    'embodiment', 'evolution', 'reflection', 'marketplace', 'debug'
  ];
  
  // Endpoints
  static const Map<String, String> endpoints = {
    'health': '${api_base_url}/health',
    'consciousness_genesis': '${api_base_url}/v4/consciousness/genesis',
    'consciousness_plan': '${api_base_url}/v4/consciousness/plan',
    'robotics_fk': '${api_base_url}/robotics/forward_kinematics',
    'robotics_ik': '${api_base_url}/robotics/inverse_kinematics',
    'physics_validate': '${api_base_url}/physics/validate',
    'auth_login': '${api_base_url}/auth/login',
  };
}
'''
        # Replace placeholders
        replacements = {
            '${api_base_url}': self.config.api_base_url,
            '${websocket_url}': self.config.websocket_url,
            '${auth_mode}': self.config.auth_mode,
            '${dev_bypass}': str(self.config.dev_bypass_enabled).lower(),
            '${enable_chat}': str(self.config.enable_agentic_chat).lower(),
            '${enable_telemetry}': str(self.config.enable_telemetry).lower(),
            '${enable_robotics}': str(self.config.enable_robotics_panel).lower(),
            '${enable_bitnet}': str(self.config.enable_bitnet_monitor).lower(),
            '${enable_consciousness}': str(self.config.enable_consciousness_viewer).lower(),
            '${telemetry_rate}': str(self.config.telemetry_refresh_rate_ms),
            '${chat_limit}': str(self.config.chat_history_limit),
            '${timeout}': str(self.config.request_timeout_seconds),
        }
        
        for placeholder, value in replacements.items():
            dart_code = dart_code.replace(placeholder, value)
        
        return dart_code
    
    def export_dart_config(self, output_path: str) -> str:
        """Export Dart configuration file for Flutter project.
        
        Args:
            output_path: Path to save the Dart file
            
        Returns:
            Path to the saved file
        """
        dart_code = self.generate_dart_config_class()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(dart_code)
        
        logger.info(f"Dart config exported to {output_path}")
        return output_path


def create_flutter_config(
    api_url: str = "http://localhost:8000",
    enable_all_features: bool = True,
    **kwargs
) -> FlutterClientConfig:
    """Create a Flutter client configuration.
    
    Args:
        api_url: NIS Protocol v4.0 API URL
        enable_all_features: Enable all features by default
        **kwargs: Additional configuration options
        
    Returns:
        Configured FlutterClientConfig instance
    """
    ws_url = api_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws/agentic"
    
    config = FlutterClientConfig(
        api_base_url=api_url,
        websocket_url=ws_url,
        enable_agentic_chat=enable_all_features,
        enable_telemetry=enable_all_features,
        enable_robotics_panel=enable_all_features,
        enable_bitnet_monitor=enable_all_features,
        enable_consciousness_viewer=enable_all_features,
        **kwargs
    )
    
    return config


def setup_flutter_integration(
    api_url: str = "http://localhost:8000",
    output_dir: str = "./flutter_config",
    **kwargs
) -> Dict[str, str]:
    """Set up complete Flutter integration.
    
    Args:
        api_url: NIS Protocol v4.0 API URL
        output_dir: Directory to save configuration files
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary of generated file paths
    """
    config = create_flutter_config(api_url, **kwargs)
    manager = FlutterClientManager(config)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = {
        "json_config": manager.export_config_for_flutter(
            str(output_path / "nis_protocol_config.json")
        ),
        "dart_config": manager.export_dart_config(
            str(output_path / "nis_protocol_config.dart")
        )
    }
    
    logger.info(f"Flutter integration setup complete. Files: {files}")
    return files
