"""
NIS Protocol Adapters

This module provides adapter classes for integrating external protocols with NIS Protocol.

Supported Adapters:
- NISv4Adapter: Integration with NIS Protocol v4.0 (10-phase consciousness, robotics, physics)
- MCPAdapter: Model Context Protocol integration
- ACPAdapter: Agent Communication Protocol integration
- A2AAdapter: Agent2Agent Protocol integration
"""

from .base_adapter import BaseAdapter
from .nis_v4_adapter import (
    NISv4Adapter,
    NISv4Config,
    ConsciousnessPhase,
    RoboticsCapability,
    create_nis_v4_adapter
)
from .mcp_adapter import MCPAdapter
from .flutter_client_config import (
    FlutterClientConfig,
    FlutterClientManager,
    AgenticChatConfig,
    TelemetryConfig,
    RoboticsControlConfig,
    create_flutter_config,
    setup_flutter_integration
)
from .neurolinux_adapter import (
    NeuroLinuxAdapter,
    NeuroLinuxConfig,
    NeuroLinuxServiceStatus,
    NeuroLinuxPhase,
    EdgeDevice,
    NeuroLinuxService,
    create_neurolinux_adapter,
    quick_connect as neurolinux_quick_connect
)

__all__ = [
    # Base
    "BaseAdapter",
    # NIS v4.0 Adapter
    "NISv4Adapter",
    "NISv4Config",
    "ConsciousnessPhase",
    "RoboticsCapability",
    "create_nis_v4_adapter",
    # MCP Adapter
    "MCPAdapter",
    # Flutter Client
    "FlutterClientConfig",
    "FlutterClientManager",
    "AgenticChatConfig",
    "TelemetryConfig",
    "RoboticsControlConfig",
    "create_flutter_config",
    "setup_flutter_integration",
    # NeuroLinux Adapter
    "NeuroLinuxAdapter",
    "NeuroLinuxConfig",
    "NeuroLinuxServiceStatus",
    "NeuroLinuxPhase",
    "EdgeDevice",
    "NeuroLinuxService",
    "create_neurolinux_adapter",
    "neurolinux_quick_connect",
] 