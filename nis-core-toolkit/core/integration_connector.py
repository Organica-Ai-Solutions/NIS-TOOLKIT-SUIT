#!/usr/bin/env python3
"""
NIS Integration Connector
Real integration system for connecting to actual NIS projects
"""

import asyncio
import json
import aiohttp
import websockets
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path
import hashlib
import hmac
import base64

class IntegrationStatus(Enum):
    """Integration connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    AUTHENTICATED = "authenticated"

class MessageType(Enum):
    """Message types for NIS integration"""
    SYSTEM_STATUS = "system_status"
    AGENT_COMMAND = "agent_command"
    DATA_SYNC = "data_sync"
    MISSION_COORDINATE = "mission_coordinate"
    HEALTH_CHECK = "health_check"
    AUTHENTICATION = "authentication"
    NOTIFICATION = "notification"

@dataclass
class NISProject:
    """NIS project connection information"""
    project_id: str
    project_name: str
    project_type: str
    endpoint_url: str
    websocket_url: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    integration_version: str = "1.0"
    status: IntegrationStatus = IntegrationStatus.DISCONNECTED
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_type": self.project_type,
            "endpoint_url": self.endpoint_url,
            "websocket_url": self.websocket_url,
            "capabilities": self.capabilities,
            "integration_version": self.integration_version,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "metadata": self.metadata
        }

@dataclass
class IntegrationMessage:
    """Message format for NIS integration"""
    message_id: str
    message_type: MessageType
    source_project: str
    target_project: str
    payload: Dict[str, Any]
    timestamp: datetime
    signature: Optional[str] = None
    priority: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source_project": self.source_project,
            "target_project": self.target_project,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "signature": self.signature,
            "priority": self.priority
        }

class NISIntegrationConnector:
    """
    Real NIS Integration Connector
    
    Provides standardized connectivity to NIS projects:
    - NIS-X (Space Systems)
    - NIS-DRONE (Hardware Systems)
    - Archaeological Research
    - SPARKNOVA (Development Platform)
    - ORION (LLM Integration)
    - SPARKNOCA (Analytics)
    - NIS-HUB (Orchestration)
    """
    
    def __init__(self, connector_id: str, config: Dict[str, Any]):
        self.connector_id = connector_id
        self.config = config
        self.connected_projects: Dict[str, NISProject] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.websocket_connections: Dict[str, Any] = {}
        self.message_queue: List[IntegrationMessage] = []
        self.integration_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_connections = config.get("max_connections", 20)
        self.message_timeout = config.get("message_timeout", 30)  # seconds
        self.heartbeat_interval = config.get("heartbeat_interval", 60)  # seconds
        self.retry_attempts = config.get("retry_attempts", 3)
        self.authentication_enabled = config.get("authentication_enabled", True)
        
        # Initialize systems
        self.logger = logging.getLogger(__name__)
        self._initialize_integration_systems()
        self._load_project_registry()
    
    def _initialize_integration_systems(self):
        """Initialize integration systems"""
        
        # Message routing system
        self.message_router = {
            "routing_table": {},
            "message_queue": [],
            "delivery_confirmation": {},
            "status": "operational"
        }
        
        # Authentication system
        self.auth_system = {
            "enabled": self.authentication_enabled,
            "key_management": {},
            "session_tokens": {},
            "access_control": {},
            "status": "operational"
        }
        
        # Monitoring system
        self.monitoring = {
            "connection_health": {},
            "message_throughput": {},
            "error_tracking": {},
            "performance_metrics": {},
            "status": "operational"
        }
    
    def _load_project_registry(self):
        """Load NIS project registry"""
        
        # NIS-X Space Systems
        self.project_registry = {
            "nis-x": {
                "project_name": "NIS-X Space Systems",
                "project_type": "space_systems",
                "default_endpoint": "https://api.nis-x.space/v1",
                "default_websocket": "wss://ws.nis-x.space/v1",
                "capabilities": [
                    "orbital_navigation",
                    "mission_planning",
                    "spacecraft_control",
                    "telemetry_streaming",
                    "trajectory_optimization"
                ],
                "required_auth": True,
                "api_version": "2.0"
            },
            "nis-drone": {
                "project_name": "NIS-DRONE Swarm Systems",
                "project_type": "hardware_systems",
                "default_endpoint": "https://api.nis-drone.com/v1",
                "default_websocket": "wss://ws.nis-drone.com/v1",
                "capabilities": [
                    "swarm_coordination",
                    "mission_execution",
                    "drone_control",
                    "formation_flight",
                    "emergency_response"
                ],
                "required_auth": True,
                "api_version": "1.5"
            },
            "archaeological-research": {
                "project_name": "Archaeological Heritage Research",
                "project_type": "cultural_systems",
                "default_endpoint": "https://api.archaeological-research.org/v1",
                "default_websocket": "wss://ws.archaeological-research.org/v1",
                "capabilities": [
                    "site_documentation",
                    "artifact_management",
                    "community_engagement",
                    "cultural_preservation",
                    "ethical_compliance"
                ],
                "required_auth": True,
                "api_version": "1.0"
            },
            "sparknova": {
                "project_name": "SPARKNOVA Development Platform",
                "project_type": "development_platform",
                "default_endpoint": "https://api.sparknova.dev/v1",
                "default_websocket": "wss://ws.sparknova.dev/v1",
                "capabilities": [
                    "development_tools",
                    "deployment_automation",
                    "code_generation",
                    "testing_frameworks",
                    "monitoring_tools"
                ],
                "required_auth": True,
                "api_version": "2.1"
            },
            "orion": {
                "project_name": "ORION LLM Integration",
                "project_type": "llm_integration",
                "default_endpoint": "https://api.orion-llm.com/v1",
                "default_websocket": "wss://ws.orion-llm.com/v1",
                "capabilities": [
                    "natural_language_processing",
                    "conversation_management",
                    "knowledge_integration",
                    "reasoning_support",
                    "multi_modal_processing"
                ],
                "required_auth": True,
                "api_version": "1.3"
            },
            "sparknoca": {
                "project_name": "SPARKNOCA Analytics Platform",
                "project_type": "analytics_platform",
                "default_endpoint": "https://api.sparknoca.analytics/v1",
                "default_websocket": "wss://ws.sparknoca.analytics/v1",
                "capabilities": [
                    "data_analytics",
                    "performance_monitoring",
                    "predictive_analytics",
                    "visualization_tools",
                    "reporting_systems"
                ],
                "required_auth": True,
                "api_version": "2.0"
            },
            "nis-hub": {
                "project_name": "NIS-HUB Orchestration",
                "project_type": "orchestration",
                "default_endpoint": "https://api.nis-hub.org/v1",
                "default_websocket": "wss://ws.nis-hub.org/v1",
                "capabilities": [
                    "system_orchestration",
                    "workflow_management",
                    "resource_coordination",
                    "service_discovery",
                    "load_balancing"
                ],
                "required_auth": True,
                "api_version": "2.2"
            }
        }
    
    async def connect_to_project(self, project_id: str, connection_config: Dict[str, Any]) -> bool:
        """Connect to a NIS project"""
        
        if project_id not in self.project_registry:
            raise ValueError(f"Unknown project: {project_id}")
        
        if len(self.connected_projects) >= self.max_connections:
            raise ValueError(f"Maximum connections ({self.max_connections}) reached")
        
        project_info = self.project_registry[project_id]
        
        # Create project connection
        project = NISProject(
            project_id=project_id,
            project_name=project_info["project_name"],
            project_type=project_info["project_type"],
            endpoint_url=connection_config.get("endpoint_url", project_info["default_endpoint"]),
            websocket_url=connection_config.get("websocket_url", project_info["default_websocket"]),
            api_key=connection_config.get("api_key"),
            api_secret=connection_config.get("api_secret"),
            capabilities=project_info["capabilities"],
            integration_version=project_info["api_version"],
            metadata=connection_config.get("metadata", {})
        )
        
        # Attempt connection
        try:
            project.status = IntegrationStatus.CONNECTING
            
            # Test HTTP connection
            http_connected = await self._test_http_connection(project)
            if not http_connected:
                project.status = IntegrationStatus.ERROR
                return False
            
            # Authenticate if required
            if project_info["required_auth"]:
                authenticated = await self._authenticate_project(project)
                if not authenticated:
                    project.status = IntegrationStatus.ERROR
                    return False
                project.status = IntegrationStatus.AUTHENTICATED
            
            # Establish WebSocket connection
            if project.websocket_url:
                ws_connected = await self._establish_websocket_connection(project)
                if not ws_connected:
                    self.logger.warning(f"WebSocket connection failed for {project_id}, continuing with HTTP only")
            
            # Connection successful
            project.status = IntegrationStatus.CONNECTED
            project.last_heartbeat = datetime.now()
            
            # Register project
            self.connected_projects[project_id] = project
            
            # Start heartbeat monitoring
            asyncio.create_task(self._monitor_project_heartbeat(project_id))
            
            self.logger.info(f"Successfully connected to {project_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to {project_id}: {str(e)}")
            project.status = IntegrationStatus.ERROR
            return False
    
    async def _test_http_connection(self, project: NISProject) -> bool:
        """Test HTTP connection to project"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{project.endpoint_url}/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"HTTP connection test failed: {str(e)}")
            return False
    
    async def _authenticate_project(self, project: NISProject) -> bool:
        """Authenticate with project"""
        
        if not project.api_key or not project.api_secret:
            self.logger.error("API credentials required for authentication")
            return False
        
        try:
            # Create authentication payload
            timestamp = str(int(datetime.now().timestamp()))
            message = f"{project.project_id}:{timestamp}"
            signature = hmac.new(
                project.api_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            auth_payload = {
                "project_id": project.project_id,
                "api_key": project.api_key,
                "timestamp": timestamp,
                "signature": signature,
                "integration_version": project.integration_version
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{project.endpoint_url}/auth/integrate",
                    json=auth_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        auth_result = await response.json()
                        project.metadata["auth_token"] = auth_result.get("token")
                        project.metadata["auth_expires"] = auth_result.get("expires")
                        return True
                    else:
                        self.logger.error(f"Authentication failed: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return False
    
    async def _establish_websocket_connection(self, project: NISProject) -> bool:
        """Establish WebSocket connection"""
        
        try:
            # Create WebSocket connection
            headers = {}
            if project.metadata.get("auth_token"):
                headers["Authorization"] = f"Bearer {project.metadata['auth_token']}"
            
            websocket = await websockets.connect(
                project.websocket_url,
                extra_headers=headers,
                timeout=10
            )
            
            self.websocket_connections[project.project_id] = websocket
            
            # Start message listener
            asyncio.create_task(self._listen_websocket_messages(project.project_id))
            
            return True
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {str(e)}")
            return False
    
    async def _listen_websocket_messages(self, project_id: str):
        """Listen for WebSocket messages"""
        
        websocket = self.websocket_connections.get(project_id)
        if not websocket:
            return
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_incoming_message(project_id, data)
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON message from {project_id}: {message}")
                except Exception as e:
                    self.logger.error(f"Error handling message from {project_id}: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"WebSocket connection to {project_id} closed")
            await self._handle_websocket_disconnect(project_id)
        except Exception as e:
            self.logger.error(f"WebSocket error for {project_id}: {str(e)}")
            await self._handle_websocket_disconnect(project_id)
    
    async def _handle_incoming_message(self, project_id: str, message_data: Dict[str, Any]):
        """Handle incoming message from project"""
        
        try:
            message = IntegrationMessage(
                message_id=message_data.get("message_id", ""),
                message_type=MessageType(message_data.get("message_type", "notification")),
                source_project=project_id,
                target_project=self.connector_id,
                payload=message_data.get("payload", {}),
                timestamp=datetime.fromisoformat(message_data.get("timestamp", datetime.now().isoformat())),
                signature=message_data.get("signature"),
                priority=message_data.get("priority", 5)
            )
            
            # Route message to appropriate handler
            await self._route_message(message)
            
        except Exception as e:
            self.logger.error(f"Error processing message from {project_id}: {str(e)}")
    
    async def _route_message(self, message: IntegrationMessage):
        """Route message to appropriate handler"""
        
        handler_key = f"{message.message_type.value}_{message.source_project}"
        
        if handler_key in self.message_handlers:
            await self.message_handlers[handler_key](message)
        else:
            # Default handler
            await self._default_message_handler(message)
    
    async def _default_message_handler(self, message: IntegrationMessage):
        """Default message handler"""
        
        self.logger.info(f"Received message: {message.message_type.value} from {message.source_project}")
        
        # Store in integration history
        self.integration_history.append({
            "message": message.to_dict(),
            "processed_at": datetime.now().isoformat(),
            "handler": "default"
        })
    
    async def _handle_websocket_disconnect(self, project_id: str):
        """Handle WebSocket disconnection"""
        
        if project_id in self.websocket_connections:
            del self.websocket_connections[project_id]
        
        if project_id in self.connected_projects:
            self.connected_projects[project_id].status = IntegrationStatus.DISCONNECTED
            
        self.logger.warning(f"WebSocket disconnected from {project_id}")
    
    async def _monitor_project_heartbeat(self, project_id: str):
        """Monitor project heartbeat"""
        
        while project_id in self.connected_projects:
            try:
                project = self.connected_projects[project_id]
                
                # Send heartbeat
                heartbeat_result = await self._send_heartbeat(project)
                
                if heartbeat_result:
                    project.last_heartbeat = datetime.now()
                    project.status = IntegrationStatus.CONNECTED
                else:
                    project.status = IntegrationStatus.ERROR
                    self.logger.warning(f"Heartbeat failed for {project_id}")
                
                # Wait for next heartbeat
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitoring error for {project_id}: {str(e)}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self, project: NISProject) -> bool:
        """Send heartbeat to project"""
        
        try:
            heartbeat_payload = {
                "connector_id": self.connector_id,
                "timestamp": datetime.now().isoformat(),
                "status": "healthy"
            }
            
            async with aiohttp.ClientSession() as session:
                headers = {}
                if project.metadata.get("auth_token"):
                    headers["Authorization"] = f"Bearer {project.metadata['auth_token']}"
                
                async with session.post(
                    f"{project.endpoint_url}/heartbeat",
                    json=heartbeat_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"Heartbeat error: {str(e)}")
            return False
    
    async def send_message(self, target_project: str, message_type: MessageType, 
                          payload: Dict[str, Any], priority: int = 5) -> bool:
        """Send message to target project"""
        
        if target_project not in self.connected_projects:
            raise ValueError(f"Not connected to project: {target_project}")
        
        project = self.connected_projects[target_project]
        
        # Create message
        message = IntegrationMessage(
            message_id=f"{self.connector_id}_{int(datetime.now().timestamp())}",
            message_type=message_type,
            source_project=self.connector_id,
            target_project=target_project,
            payload=payload,
            timestamp=datetime.now(),
            priority=priority
        )
        
        # Sign message if authentication is enabled
        if self.authentication_enabled and project.api_secret:
            message.signature = self._sign_message(message, project.api_secret)
        
        # Send via WebSocket if available, otherwise HTTP
        if target_project in self.websocket_connections:
            return await self._send_websocket_message(target_project, message)
        else:
            return await self._send_http_message(target_project, message)
    
    def _sign_message(self, message: IntegrationMessage, secret: str) -> str:
        """Sign message for authentication"""
        
        message_string = f"{message.message_id}:{message.timestamp.isoformat()}:{json.dumps(message.payload, sort_keys=True)}"
        signature = hmac.new(
            secret.encode(),
            message_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    async def _send_websocket_message(self, target_project: str, message: IntegrationMessage) -> bool:
        """Send message via WebSocket"""
        
        try:
            websocket = self.websocket_connections[target_project]
            await websocket.send(json.dumps(message.to_dict()))
            return True
            
        except Exception as e:
            self.logger.error(f"WebSocket message send error: {str(e)}")
            return False
    
    async def _send_http_message(self, target_project: str, message: IntegrationMessage) -> bool:
        """Send message via HTTP"""
        
        try:
            project = self.connected_projects[target_project]
            
            headers = {"Content-Type": "application/json"}
            if project.metadata.get("auth_token"):
                headers["Authorization"] = f"Bearer {project.metadata['auth_token']}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{project.endpoint_url}/messages",
                    json=message.to_dict(),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.message_timeout)
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"HTTP message send error: {str(e)}")
            return False
    
    async def coordinate_mission(self, mission_data: Dict[str, Any], 
                               participating_projects: List[str]) -> Dict[str, Any]:
        """Coordinate mission across multiple NIS projects"""
        
        coordination_results = {}
        
        # Send mission coordination message to all participants
        for project_id in participating_projects:
            if project_id in self.connected_projects:
                result = await self.send_message(
                    target_project=project_id,
                    message_type=MessageType.MISSION_COORDINATE,
                    payload={
                        "mission_id": mission_data.get("mission_id"),
                        "mission_type": mission_data.get("mission_type"),
                        "coordination_role": mission_data.get("roles", {}).get(project_id, "participant"),
                        "mission_parameters": mission_data.get("parameters", {}),
                        "timeline": mission_data.get("timeline", {}),
                        "coordination_requirements": mission_data.get("coordination_requirements", {})
                    },
                    priority=8
                )
                
                coordination_results[project_id] = {
                    "coordination_sent": result,
                    "timestamp": datetime.now().isoformat()
                }
        
        return {
            "mission_id": mission_data.get("mission_id"),
            "coordination_results": coordination_results,
            "participating_projects": participating_projects,
            "coordination_timestamp": datetime.now().isoformat()
        }
    
    async def sync_data(self, source_project: str, target_projects: List[str], 
                       data_type: str, data_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize data between NIS projects"""
        
        sync_results = {}
        
        for target_project in target_projects:
            if target_project in self.connected_projects:
                result = await self.send_message(
                    target_project=target_project,
                    message_type=MessageType.DATA_SYNC,
                    payload={
                        "source_project": source_project,
                        "data_type": data_type,
                        "data_payload": data_payload,
                        "sync_timestamp": datetime.now().isoformat(),
                        "sync_version": "1.0"
                    },
                    priority=6
                )
                
                sync_results[target_project] = {
                    "sync_sent": result,
                    "timestamp": datetime.now().isoformat()
                }
        
        return {
            "source_project": source_project,
            "data_type": data_type,
            "sync_results": sync_results,
            "sync_timestamp": datetime.now().isoformat()
        }
    
    def register_message_handler(self, message_type: MessageType, source_project: str, 
                                handler: Callable[[IntegrationMessage], None]):
        """Register custom message handler"""
        
        handler_key = f"{message_type.value}_{source_project}"
        self.message_handlers[handler_key] = handler
    
    async def disconnect_project(self, project_id: str) -> bool:
        """Disconnect from project"""
        
        if project_id not in self.connected_projects:
            return False
        
        # Close WebSocket connection
        if project_id in self.websocket_connections:
            await self.websocket_connections[project_id].close()
            del self.websocket_connections[project_id]
        
        # Remove from connected projects
        del self.connected_projects[project_id]
        
        self.logger.info(f"Disconnected from {project_id}")
        return True
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        
        connection_stats = {
            "total_connections": len(self.connected_projects),
            "connected_projects": list(self.connected_projects.keys()),
            "websocket_connections": len(self.websocket_connections),
            "message_handlers": len(self.message_handlers)
        }
        
        project_status = {}
        for project_id, project in self.connected_projects.items():
            project_status[project_id] = {
                "status": project.status.value,
                "last_heartbeat": project.last_heartbeat.isoformat() if project.last_heartbeat else None,
                "capabilities": project.capabilities,
                "integration_version": project.integration_version
            }
        
        return {
            "connector_id": self.connector_id,
            "timestamp": datetime.now().isoformat(),
            "connection_statistics": connection_stats,
            "project_status": project_status,
            "integration_history_count": len(self.integration_history),
            "system_health": {
                "message_router": self.message_router["status"],
                "auth_system": self.auth_system["status"],
                "monitoring": self.monitoring["status"]
            }
        }

# Factory function for creating integration connector
def create_nis_integration_connector(connector_id: str, config: Dict[str, Any]) -> NISIntegrationConnector:
    """Create NIS integration connector"""
    
    return NISIntegrationConnector(connector_id, config)

# Example usage
async def example_nis_integration():
    """Example NIS project integration"""
    
    # Create integration connector
    connector = create_nis_integration_connector(
        connector_id="NIS-TOOLKIT-CONNECTOR",
        config={
            "max_connections": 10,
            "authentication_enabled": True,
            "heartbeat_interval": 30
        }
    )
    
    # Connect to NIS-X
    await connector.connect_to_project("nis-x", {
        "endpoint_url": "https://api.nis-x.space/v1",
        "api_key": "your-api-key",
        "api_secret": "your-api-secret"
    })
    
    # Connect to NIS-DRONE
    await connector.connect_to_project("nis-drone", {
        "endpoint_url": "https://api.nis-drone.com/v1",
        "api_key": "your-api-key",
        "api_secret": "your-api-secret"
    })
    
    # Coordinate joint mission
    mission_result = await connector.coordinate_mission({
        "mission_id": "joint_mission_001",
        "mission_type": "surveillance_support",
        "roles": {
            "nis-x": "orbital_surveillance",
            "nis-drone": "ground_surveillance"
        },
        "parameters": {
            "area_of_interest": {"lat": 45.0, "lon": -75.0, "radius": 10000},
            "duration": 7200,
            "coordination_frequency": 300
        }
    }, ["nis-x", "nis-drone"])
    
    print(f"Mission coordination result: {mission_result}")
    
    # Get integration status
    status = await connector.get_integration_status()
    print(f"Integration status: {status}")

if __name__ == "__main__":
    asyncio.run(example_nis_integration()) 