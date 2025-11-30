"""
NeuroLinux Adapter for NIS-TOOLKIT-SUIT

This module provides integration between NIS-TOOLKIT-SUIT and NeuroLinux,
the cognitive operating system for edge robotics and distributed AI.

NeuroLinux Components:
- NeuroGrid: Distributed mesh network for device coordination
- NeuroHub: Cloud controller for system management
- NeuroForge: SDK and development tools
- NeuroStore: Package registry for AI components
- NeuroKernel: Rust-based agent scheduling kernel

Features:
- Service orchestration via NIS Bridge API
- Edge device deployment
- Distributed agent coordination
- Real-time telemetry from edge devices
- OTA updates for AI models
"""

import asyncio
import logging
import json
import time
import aiohttp
import requests
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class NeuroLinuxServiceStatus(Enum):
    """Service status for NeuroLinux services"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"


class NeuroLinuxPhase(Enum):
    """NeuroLinux development phases"""
    PHASE0_FOUNDATION = "phase0"      # NIS Protocol Foundation
    PHASE1_RUNTIME = "phase1"         # NeuroLinux Runtime
    PHASE2_NEUROKERNEL = "phase2"     # NeuroKernel Layer
    PHASE3_SYSTEM = "phase3"          # Full OS Integration
    PHASE4_DISTRIBUTED = "phase4"     # Distributed Ecosystem


@dataclass
class NeuroLinuxConfig:
    """Configuration for NeuroLinux connection"""
    # Bridge API settings
    bridge_url: str = "http://localhost:8080"
    bridge_timeout: int = 30
    
    # NeuroHub settings (cloud controller)
    neurohub_url: str = "http://localhost:9000"
    neurohub_api_key: Optional[str] = None
    
    # NeuroGrid settings (mesh network)
    neurogrid_enabled: bool = True
    neurogrid_discovery_port: int = 5353
    
    # Device settings
    device_id: Optional[str] = None
    device_type: str = "development"  # development, edge, cloud
    
    # Features
    enable_telemetry: bool = True
    enable_ota_updates: bool = True
    enable_federation: bool = False
    
    # Paths
    neurolinux_path: Optional[str] = None


@dataclass
class EdgeDevice:
    """Represents an edge device running NeuroLinux"""
    device_id: str
    hostname: str
    ip_address: str
    device_type: str
    status: NeuroLinuxServiceStatus
    capabilities: List[str] = field(default_factory=list)
    agents_running: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: float = 0.0


@dataclass
class NeuroLinuxService:
    """Represents a service running on NeuroLinux"""
    name: str
    status: NeuroLinuxServiceStatus
    pid: Optional[int] = None
    uptime: Optional[float] = None
    endpoint: Optional[str] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)


class NeuroLinuxAdapter:
    """
    Adapter for integrating NIS-TOOLKIT-SUIT with NeuroLinux.
    
    Provides:
    - Connection to NIS Bridge API (Phase 0)
    - Service orchestration
    - Edge device management
    - Distributed agent deployment
    - Telemetry collection
    - OTA model updates
    """
    
    def __init__(self, config: Optional[NeuroLinuxConfig] = None):
        """Initialize the NeuroLinux adapter.
        
        Args:
            config: NeuroLinux configuration
        """
        self.config = config or NeuroLinuxConfig()
        self.session = requests.Session()
        self._async_session: Optional[aiohttp.ClientSession] = None
        
        # State
        self.connected = False
        self.services: Dict[str, NeuroLinuxService] = {}
        self.devices: Dict[str, EdgeDevice] = {}
        self.ipc_handlers: Dict[str, Callable] = {}
        
        # Setup session headers
        if self.config.neurohub_api_key:
            self.session.headers["X-API-Key"] = self.config.neurohub_api_key
        
        logger.info(f"NeuroLinux adapter initialized for {self.config.bridge_url}")
    
    # ==================== Connection Management ====================
    
    async def connect(self) -> bool:
        """Connect to NeuroLinux Bridge API.
        
        Returns:
            True if connection successful
        """
        try:
            # Test bridge connection
            response = self.session.get(
                f"{self.config.bridge_url}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                self.connected = True
                logger.info("Connected to NeuroLinux Bridge API")
                
                # Discover services
                await self.discover_services()
                
                return True
            else:
                logger.warning(f"Bridge health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to NeuroLinux: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from NeuroLinux."""
        self.connected = False
        self.services.clear()
        logger.info("Disconnected from NeuroLinux")
    
    # ==================== Service Management ====================
    
    async def discover_services(self) -> Dict[str, NeuroLinuxService]:
        """Discover running NeuroLinux services.
        
        Returns:
            Dictionary of discovered services
        """
        try:
            response = self.session.get(
                f"{self.config.bridge_url}/services",
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            services_data = response.json()
            
            for name, info in services_data.get("services", {}).items():
                self.services[name] = NeuroLinuxService(
                    name=name,
                    status=NeuroLinuxServiceStatus(info.get("status", "unknown")),
                    pid=info.get("pid"),
                    uptime=info.get("uptime"),
                    endpoint=info.get("endpoint"),
                    resource_usage=info.get("resource_usage", {})
                )
            
            logger.info(f"Discovered {len(self.services)} services")
            return self.services
            
        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            return {}
    
    async def start_service(self, service_name: str, **kwargs) -> Dict[str, Any]:
        """Start a NeuroLinux service.
        
        Args:
            service_name: Name of the service to start
            **kwargs: Additional service parameters
            
        Returns:
            Service start result
        """
        try:
            response = self.session.post(
                f"{self.config.bridge_url}/services/{service_name}/start",
                json=kwargs,
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Update local state
            if result.get("success"):
                self.services[service_name] = NeuroLinuxService(
                    name=service_name,
                    status=NeuroLinuxServiceStatus.RUNNING,
                    pid=result.get("pid")
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to start service {service_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def stop_service(self, service_name: str, force: bool = False) -> Dict[str, Any]:
        """Stop a NeuroLinux service.
        
        Args:
            service_name: Name of the service to stop
            force: Force stop (SIGKILL instead of SIGTERM)
            
        Returns:
            Service stop result
        """
        try:
            response = self.session.post(
                f"{self.config.bridge_url}/services/{service_name}/stop",
                json={"force": force},
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Update local state
            if result.get("success"):
                if service_name in self.services:
                    self.services[service_name].status = NeuroLinuxServiceStatus.STOPPED
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to stop service {service_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status of a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service status information
        """
        try:
            response = self.session.get(
                f"{self.config.bridge_url}/services/{service_name}/status",
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get status for {service_name}: {e}")
            return {"error": str(e)}
    
    # ==================== Agent Deployment ====================
    
    async def deploy_agent(
        self,
        agent_id: str,
        agent_type: str,
        config: Dict[str, Any],
        target_device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Deploy an agent to NeuroLinux.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (consciousness, robotics, etc.)
            config: Agent configuration
            target_device: Target device ID (None for local)
            
        Returns:
            Deployment result
        """
        try:
            payload = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "config": config,
                "target_device": target_device
            }
            
            response = self.session.post(
                f"{self.config.bridge_url}/agents/deploy",
                json=payload,
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Agent deployment failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def list_agents(self, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List deployed agents.
        
        Args:
            device_id: Filter by device (None for all)
            
        Returns:
            List of deployed agents
        """
        try:
            params = {}
            if device_id:
                params["device_id"] = device_id
            
            response = self.session.get(
                f"{self.config.bridge_url}/agents",
                params=params,
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            return response.json().get("agents", [])
            
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return []
    
    async def send_to_agent(
        self,
        agent_id: str,
        message: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Send a message to an agent via IPC.
        
        Args:
            agent_id: Target agent ID
            message: Message to send
            timeout: Response timeout
            
        Returns:
            Agent response
        """
        try:
            payload = {
                "agent_id": agent_id,
                "message": message,
                "timeout": timeout
            }
            
            response = self.session.post(
                f"{self.config.bridge_url}/ipc/send",
                json=payload,
                timeout=timeout + 5
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"IPC send failed: {e}")
            return {"error": str(e)}
    
    # ==================== Edge Device Management ====================
    
    async def discover_devices(self) -> Dict[str, EdgeDevice]:
        """Discover edge devices on the NeuroGrid.
        
        Returns:
            Dictionary of discovered devices
        """
        if not self.config.neurogrid_enabled:
            logger.warning("NeuroGrid is disabled")
            return {}
        
        try:
            response = self.session.get(
                f"{self.config.neurohub_url}/api/devices",
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            devices_data = response.json()
            
            for device_info in devices_data.get("devices", []):
                device = EdgeDevice(
                    device_id=device_info["device_id"],
                    hostname=device_info.get("hostname", "unknown"),
                    ip_address=device_info.get("ip_address", ""),
                    device_type=device_info.get("device_type", "unknown"),
                    status=NeuroLinuxServiceStatus(device_info.get("status", "unknown")),
                    capabilities=device_info.get("capabilities", []),
                    agents_running=device_info.get("agents", []),
                    resources=device_info.get("resources", {}),
                    last_heartbeat=device_info.get("last_heartbeat", 0)
                )
                self.devices[device.device_id] = device
            
            logger.info(f"Discovered {len(self.devices)} edge devices")
            return self.devices
            
        except Exception as e:
            logger.error(f"Device discovery failed: {e}")
            return {}
    
    async def get_device_telemetry(self, device_id: str) -> Dict[str, Any]:
        """Get telemetry data from an edge device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Telemetry data
        """
        try:
            response = self.session.get(
                f"{self.config.neurohub_url}/api/devices/{device_id}/telemetry",
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get telemetry for {device_id}: {e}")
            return {"error": str(e)}
    
    async def execute_on_device(
        self,
        device_id: str,
        command: str,
        args: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute a command on an edge device.
        
        Args:
            device_id: Target device
            command: Command to execute
            args: Command arguments
            
        Returns:
            Execution result
        """
        try:
            payload = {
                "command": command,
                "args": args or []
            }
            
            response = self.session.post(
                f"{self.config.neurohub_url}/api/devices/{device_id}/execute",
                json=payload,
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Remote execution failed: {e}")
            return {"error": str(e)}
    
    # ==================== NeuroKernel Integration ====================
    
    async def get_kernel_status(self) -> Dict[str, Any]:
        """Get NeuroKernel status.
        
        Returns:
            Kernel status information
        """
        try:
            response = self.session.get(
                f"{self.config.bridge_url}/kernel/status",
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get kernel status: {e}")
            return {"error": str(e)}
    
    async def schedule_task(
        self,
        task_id: str,
        task_type: str,
        priority: int = 5,
        deadline_ms: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Schedule a task on the NeuroKernel.
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task
            priority: Task priority (1-10, higher = more important)
            deadline_ms: Soft deadline in milliseconds
            payload: Task payload
            
        Returns:
            Scheduling result
        """
        try:
            task_payload = {
                "task_id": task_id,
                "task_type": task_type,
                "priority": priority,
                "deadline_ms": deadline_ms,
                "payload": payload or {}
            }
            
            response = self.session.post(
                f"{self.config.bridge_url}/kernel/schedule",
                json=task_payload,
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Task scheduling failed: {e}")
            return {"error": str(e)}
    
    # ==================== OTA Updates ====================
    
    async def check_updates(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Check for available OTA updates.
        
        Args:
            device_id: Specific device (None for all)
            
        Returns:
            Available updates
        """
        if not self.config.enable_ota_updates:
            return {"updates": [], "message": "OTA updates disabled"}
        
        try:
            params = {}
            if device_id:
                params["device_id"] = device_id
            
            response = self.session.get(
                f"{self.config.neurohub_url}/api/updates/check",
                params=params,
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            return {"error": str(e)}
    
    async def deploy_model_update(
        self,
        model_id: str,
        version: str,
        target_devices: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Deploy a model update to devices.
        
        Args:
            model_id: Model identifier
            version: Target version
            target_devices: List of device IDs (None for all)
            
        Returns:
            Deployment result
        """
        try:
            payload = {
                "model_id": model_id,
                "version": version,
                "target_devices": target_devices
            }
            
            response = self.session.post(
                f"{self.config.neurohub_url}/api/updates/deploy",
                json=payload,
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return {"error": str(e)}
    
    # ==================== Telemetry & Monitoring ====================
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics from NeuroLinux.
        
        Returns:
            System metrics
        """
        try:
            response = self.session.get(
                f"{self.config.bridge_url}/metrics",
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {"error": str(e)}
    
    async def stream_telemetry(
        self,
        callback: Callable[[Dict[str, Any]], None],
        device_id: Optional[str] = None
    ):
        """Stream telemetry data via WebSocket.
        
        Args:
            callback: Function to call with telemetry data
            device_id: Specific device to stream from
        """
        ws_url = self.config.bridge_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/telemetry"
        
        if device_id:
            ws_url = f"{ws_url}?device_id={device_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            callback(data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            break
        except Exception as e:
            logger.error(f"Telemetry stream error: {e}")
    
    # ==================== Federation ====================
    
    async def join_federation(self, hub_url: str, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Join a NeuroHub federation.
        
        Args:
            hub_url: Federation hub URL
            credentials: Authentication credentials
            
        Returns:
            Federation join result
        """
        if not self.config.enable_federation:
            return {"success": False, "message": "Federation disabled"}
        
        try:
            payload = {
                "hub_url": hub_url,
                "credentials": credentials,
                "device_id": self.config.device_id
            }
            
            response = self.session.post(
                f"{self.config.neurohub_url}/api/federation/join",
                json=payload,
                timeout=self.config.bridge_timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Federation join failed: {e}")
            return {"error": str(e)}
    
    # ==================== Utility Methods ====================
    
    def get_status(self) -> Dict[str, Any]:
        """Get adapter status.
        
        Returns:
            Adapter status information
        """
        return {
            "connected": self.connected,
            "bridge_url": self.config.bridge_url,
            "neurohub_url": self.config.neurohub_url,
            "services_count": len(self.services),
            "devices_count": len(self.devices),
            "neurogrid_enabled": self.config.neurogrid_enabled,
            "ota_enabled": self.config.enable_ota_updates,
            "federation_enabled": self.config.enable_federation,
            "device_type": self.config.device_type
        }
    
    def get_services(self) -> Dict[str, Dict[str, Any]]:
        """Get all discovered services.
        
        Returns:
            Dictionary of services
        """
        return {
            name: {
                "name": svc.name,
                "status": svc.status.value,
                "pid": svc.pid,
                "uptime": svc.uptime,
                "endpoint": svc.endpoint,
                "resource_usage": svc.resource_usage
            }
            for name, svc in self.services.items()
        }
    
    def get_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get all discovered devices.
        
        Returns:
            Dictionary of devices
        """
        return {
            device_id: {
                "device_id": dev.device_id,
                "hostname": dev.hostname,
                "ip_address": dev.ip_address,
                "device_type": dev.device_type,
                "status": dev.status.value,
                "capabilities": dev.capabilities,
                "agents_running": dev.agents_running,
                "resources": dev.resources,
                "last_heartbeat": dev.last_heartbeat
            }
            for device_id, dev in self.devices.items()
        }


# ==================== Convenience Functions ====================

def create_neurolinux_adapter(
    bridge_url: str = "http://localhost:8080",
    neurohub_url: str = "http://localhost:9000",
    **kwargs
) -> NeuroLinuxAdapter:
    """Create a configured NeuroLinux adapter.
    
    Args:
        bridge_url: NIS Bridge API URL
        neurohub_url: NeuroHub URL
        **kwargs: Additional configuration options
        
    Returns:
        Configured NeuroLinuxAdapter instance
    """
    config = NeuroLinuxConfig(
        bridge_url=bridge_url,
        neurohub_url=neurohub_url,
        **kwargs
    )
    return NeuroLinuxAdapter(config)


async def quick_connect(
    bridge_url: str = "http://localhost:8080"
) -> Optional[NeuroLinuxAdapter]:
    """Quick connect to NeuroLinux.
    
    Args:
        bridge_url: NIS Bridge API URL
        
    Returns:
        Connected adapter or None if failed
    """
    adapter = create_neurolinux_adapter(bridge_url=bridge_url)
    
    if await adapter.connect():
        return adapter
    
    return None
