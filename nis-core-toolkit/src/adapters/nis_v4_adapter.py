"""
NIS Protocol v4.0 Adapter

This module provides the adapter for integrating with NIS Protocol v4.0,
supporting the new 10-phase consciousness pipeline, robotics integration,
BitNet local AI, and Flutter desktop client communication.

Features:
- 10-Phase Consciousness Pipeline (Genesis → Debug)
- Robotics Integration (FK/IK, trajectory planning, TMR)
- Physics Validation (PINN, Laplace, KAN)
- Authentication System (JWT, API keys)
- WebSocket streaming support
- Flutter desktop client compatibility
"""

import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
import aiohttp
import requests

from .base_adapter import BaseAdapter


class ConsciousnessPhase(Enum):
    """10-Phase Consciousness Pipeline phases from NIS Protocol v4.0"""
    GENESIS = "genesis"           # Idea generation
    PLAN = "plan"                 # Strategic planning
    COLLECTIVE = "collective"     # Multi-agent consensus
    MULTIPATH = "multipath"       # Parallel reasoning
    ETHICS = "ethics"             # Ethical evaluation
    EMBODIMENT = "embodiment"     # Physical integration
    EVOLUTION = "evolution"       # Adaptive learning
    REFLECTION = "reflection"     # Self-assessment
    MARKETPLACE = "marketplace"   # Resource allocation
    DEBUG = "debug"               # Error analysis


class RoboticsCapability(Enum):
    """Robotics capabilities in NIS Protocol v4.0"""
    FORWARD_KINEMATICS = "forward_kinematics"
    INVERSE_KINEMATICS = "inverse_kinematics"
    TRAJECTORY_PLANNING = "trajectory_planning"
    TMR_REDUNDANCY = "tmr_redundancy"
    MAVLINK = "mavlink"
    ROS = "ros"


@dataclass
class NISv4Config:
    """Configuration for NIS Protocol v4.0 connection"""
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    timeout: int = 30
    enable_websocket: bool = True
    websocket_url: Optional[str] = None
    enable_robotics: bool = True
    enable_consciousness: bool = True
    enable_physics: bool = True
    enable_bitnet: bool = False
    flutter_client_mode: bool = False
    
    def __post_init__(self):
        if self.websocket_url is None:
            # Derive WebSocket URL from base URL
            ws_scheme = "wss" if self.base_url.startswith("https") else "ws"
            host = self.base_url.replace("https://", "").replace("http://", "")
            self.websocket_url = f"{ws_scheme}://{host}"


@dataclass
class ConsciousnessRequest:
    """Request structure for consciousness pipeline"""
    phase: ConsciousnessPhase
    input_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    previous_phases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoboticsRequest:
    """Request structure for robotics operations"""
    capability: RoboticsCapability
    parameters: Dict[str, Any]
    robot_id: Optional[str] = None
    safety_constraints: Optional[Dict[str, Any]] = None


class NISv4Adapter(BaseAdapter):
    """
    Adapter for NIS Protocol v4.0 integration.
    
    Supports the full v4.0 feature set including:
    - 10-phase consciousness pipeline
    - Robotics integration with FK/IK and trajectory planning
    - Physics validation with PINN
    - Authentication with JWT and API keys
    - WebSocket streaming for real-time communication
    - Flutter desktop client compatibility
    """
    
    # API endpoint mappings for v4.0 (32 consciousness + robotics + physics + auth)
    V4_ENDPOINTS = {
        # Consciousness endpoints - Genesis Phase
        "consciousness": {
            "genesis": "/v4/consciousness/genesis",
            "genesis_history": "/v4/consciousness/genesis/history",
            
            # Plan Phase
            "plan": "/v4/consciousness/plan",
            "plan_status": "/v4/consciousness/plan/status",
            
            # Collective Phase (Multi-agent consensus)
            "collective_decide": "/v4/consciousness/collective/decide",
            "collective_register": "/v4/consciousness/collective/register",
            "collective_status": "/v4/consciousness/collective/status",
            "collective_sync": "/v4/consciousness/collective/sync",
            
            # Multipath Phase (Parallel reasoning)
            "multipath_start": "/v4/consciousness/multipath/start",
            "multipath_collapse": "/v4/consciousness/multipath/collapse",
            "multipath_state": "/v4/consciousness/multipath/state",
            
            # Ethics Phase
            "ethics": "/v4/consciousness/ethics/evaluate",
            
            # Embodiment Phase (Physical integration)
            "embodiment_status": "/v4/consciousness/embodiment/status",
            "embodiment_action": "/v4/consciousness/embodiment/action/execute",
            "embodiment_motion_check": "/v4/consciousness/embodiment/motion/check",
            "embodiment_state_update": "/v4/consciousness/embodiment/state/update",
            "embodiment_diagnostics": "/v4/consciousness/embodiment/diagnostics",
            "embodiment_redundancy_status": "/v4/consciousness/embodiment/redundancy/status",
            "embodiment_redundancy_degradation": "/v4/consciousness/embodiment/redundancy/degradation",
            "embodiment_robotics_info": "/v4/consciousness/embodiment/robotics/info",
            "embodiment_robotics_datasets": "/v4/consciousness/embodiment/robotics/datasets",
            "embodiment_vision_detect": "/v4/consciousness/embodiment/vision/detect",
            
            # Evolution Phase (Adaptive learning)
            "evolve": "/v4/consciousness/evolve",
            "evolution_history": "/v4/consciousness/evolution/history",
            "meta_evolve": "/v4/consciousness/meta-evolve",
            "meta_evolution_status": "/v4/consciousness/meta-evolution/status",
            
            # Marketplace Phase (Resource allocation)
            "marketplace_list": "/v4/consciousness/marketplace/list",
            "marketplace_publish": "/v4/consciousness/marketplace/publish",
            "marketplace_insight": "/v4/consciousness/marketplace/insight",
            
            # Debug Phase
            "debug_explain": "/v4/consciousness/debug/explain",
            "debug_record": "/v4/consciousness/debug/record",
            
            # Performance monitoring
            "performance": "/v4/consciousness/performance",
        },
        # Robotics endpoints
        "robotics": {
            "forward_kinematics": "/robotics/forward_kinematics",
            "inverse_kinematics": "/robotics/inverse_kinematics",
            "plan_trajectory": "/robotics/plan_trajectory",
            "capabilities": "/robotics/capabilities",
            "control_ws": "/ws/robotics/control",
        },
        # Physics endpoints
        "physics": {
            "validate": "/physics/validate",
            "heat_equation": "/physics/solve/heat-equation",
            "wave_equation": "/physics/solve/wave-equation",
            "constants": "/physics/constants",
        },
        # Auth endpoints
        "auth": {
            "signup": "/auth/signup",
            "login": "/auth/login",
            "logout": "/auth/logout",
            "verify": "/auth/verify",
            "refresh": "/auth/refresh",
            "api_keys": "/users/api-keys",
            "api_keys_list": "/users/api-keys/list",
            "profile": "/users/profile",
            "settings": "/users/settings",
        },
        # Vision endpoints
        "vision": {
            "analyze": "/vision/analyze",
            "generate": "/vision/generate",
            "detect": "/vision/detect",
            "ocr": "/vision/ocr",
            "describe": "/vision/describe",
        },
        # Voice endpoints
        "voice": {
            "tts": "/voice/tts",
            "stt": "/voice/stt",
            "stream": "/voice/stream",
            "voices": "/voice/voices",
        },
        # Research endpoints
        "research": {
            "search": "/research/search",
            "arxiv": "/research/arxiv",
            "analyze": "/research/analyze",
        },
        # BitNet (local AI) endpoints
        "bitnet": {
            "status": "/bitnet/status",
            "generate": "/bitnet/generate",
            "models": "/bitnet/models",
        },
        # Agents endpoints
        "agents": {
            "list": "/agents",
            "create": "/agents/create",
            "status": "/agents/status",
            "process": "/agents/process",
            "alignment_evaluate": "/agents/alignment/evaluate_ethics",
            "registry": "/agents/registry",
        },
        # Core endpoints
        "core": {
            "health": "/health",
            "docs": "/docs",
            "console": "/console",
            "process": "/process",
            "metrics": "/metrics",
            "openapi": "/openapi.json",
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the NIS v4.0 adapter.
        
        Args:
            config: Configuration dictionary or NISv4Config instance
        """
        # Set up default config structure for base adapter
        default_config = {
            "protocol_name": "NIS_v4",
            "endpoint": "http://localhost:8000",
            "timeout": 30,
            "adapter_id": "nis_v4_adapter"
        }
        
        if config:
            default_config.update(config)
        
        # Initialize base adapter with required fields
        self.config = default_config
        self.logger = logging.getLogger("nis.adapter.NISv4Adapter")
        self.connection_status = "disconnected"
        self.message_history: List[Dict[str, Any]] = []
        self.error_count = 0
        self.last_heartbeat = time.time()
        
        # Create v4-specific config
        if isinstance(config, NISv4Config):
            self.v4_config = config
        else:
            self.v4_config = NISv4Config(
                base_url=default_config.get("endpoint", "http://localhost:8000"),
                api_key=default_config.get("api_key"),
                jwt_token=default_config.get("jwt_token"),
                timeout=default_config.get("timeout", 30),
            )
        
        # HTTP session for API calls
        self.session = requests.Session()
        self._setup_auth_headers()
        
        # WebSocket connection (lazy initialization)
        self._ws_connection = None
        self._ws_callbacks: Dict[str, Callable] = {}
        
        # Consciousness pipeline state
        self._pipeline_state: Dict[str, Any] = {}
        
        self.logger.info(f"NIS v4.0 Adapter initialized for {self.v4_config.base_url}")
    
    def _setup_auth_headers(self):
        """Configure authentication headers for API requests."""
        headers = {"Content-Type": "application/json"}
        
        if self.v4_config.jwt_token:
            headers["Authorization"] = f"Bearer {self.v4_config.jwt_token}"
        elif self.v4_config.api_key:
            headers["X-API-Key"] = self.v4_config.api_key
        
        self.session.headers.update(headers)
    
    def validate_config(self) -> bool:
        """Validate the adapter configuration."""
        try:
            if not self.v4_config.base_url:
                self.logger.error("Base URL is required")
                return False
            
            # Test connection to health endpoint
            response = self.session.get(
                f"{self.v4_config.base_url}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                self.connection_status = "connected"
                return True
            else:
                self.logger.warning(f"Health check returned status {response.status_code}")
                return True  # Config is valid even if server is down
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Could not connect to NIS v4.0 server: {e}")
            return True  # Config structure is valid
    
    def translate_to_nis(self, external_message: Dict[str, Any]) -> Dict[str, Any]:
        """Translate external message to NIS Protocol toolkit format.
        
        Args:
            external_message: Message from NIS Protocol v4.0 API
            
        Returns:
            Message in NIS toolkit format
        """
        try:
            # Determine message type from v4 response
            message_type = self._detect_v4_message_type(external_message)
            
            nis_message = {
                "protocol": "NIS",
                "version": "4.0",
                "timestamp": time.time(),
                "source_protocol": "NIS_v4",
                "message_id": external_message.get("id", f"v4_{int(time.time())}"),
                "message_type": message_type,
                "payload": {},
                "metadata": {
                    "v4_endpoint": external_message.get("_endpoint", "unknown"),
                    "processed_at": time.time()
                }
            }
            
            # Handle consciousness pipeline responses
            if message_type == "consciousness":
                nis_message["payload"] = self._translate_consciousness_response(external_message)
            
            # Handle robotics responses
            elif message_type == "robotics":
                nis_message["payload"] = self._translate_robotics_response(external_message)
            
            # Handle physics responses
            elif message_type == "physics":
                nis_message["payload"] = self._translate_physics_response(external_message)
            
            # Handle auth responses
            elif message_type == "auth":
                nis_message["payload"] = self._translate_auth_response(external_message)
            
            # Generic response
            else:
                nis_message["payload"] = external_message
            
            self._log_message_translation("to_nis", external_message, nis_message)
            return nis_message
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Translation to NIS failed: {e}")
            return {
                "protocol": "NIS",
                "error": True,
                "error_message": str(e),
                "original_message": external_message
            }
    
    def translate_from_nis(self, nis_message: Dict[str, Any]) -> Dict[str, Any]:
        """Translate NIS toolkit message to NIS Protocol v4.0 format.
        
        Args:
            nis_message: Message in NIS toolkit format
            
        Returns:
            Message formatted for NIS Protocol v4.0 API
        """
        try:
            message_type = nis_message.get("message_type", "generic")
            payload = nis_message.get("payload", {})
            
            v4_message = {
                "timestamp": time.time(),
                "source": "nis_toolkit",
            }
            
            # Format for consciousness endpoints
            if message_type == "consciousness":
                phase = payload.get("phase", "genesis")
                v4_message.update({
                    "input": payload.get("input", payload.get("data", "")),
                    "context": payload.get("context", {}),
                    "options": payload.get("options", {})
                })
            
            # Format for robotics endpoints
            elif message_type == "robotics":
                v4_message.update({
                    "joint_angles": payload.get("joint_angles", []),
                    "target_pose": payload.get("target_pose", {}),
                    "robot_config": payload.get("robot_config", {}),
                    "constraints": payload.get("constraints", {})
                })
            
            # Format for physics endpoints
            elif message_type == "physics":
                v4_message.update({
                    "equation_type": payload.get("equation_type", "heat"),
                    "parameters": payload.get("parameters", {}),
                    "boundary_conditions": payload.get("boundary_conditions", {}),
                    "initial_conditions": payload.get("initial_conditions", {})
                })
            
            # Generic format
            else:
                v4_message.update(payload)
            
            self._log_message_translation("from_nis", nis_message, v4_message)
            return v4_message
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Translation from NIS failed: {e}")
            return {"error": True, "message": str(e)}
    
    def send_to_external_agent(self, agent_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message to NIS Protocol v4.0 API.
        
        Args:
            agent_id: The endpoint or agent identifier
            message: The message to send
            
        Returns:
            Response from the v4.0 API
        """
        try:
            # Determine the endpoint
            endpoint = self._resolve_endpoint(agent_id)
            url = f"{self.v4_config.base_url}{endpoint}"
            
            # Translate message to v4 format
            v4_message = self.translate_from_nis(message)
            
            # Make the API request
            response = self.session.post(
                url,
                json=v4_message,
                timeout=self.v4_config.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            result["_endpoint"] = endpoint
            
            self.connection_status = "active"
            self.last_heartbeat = time.time()
            
            return self.translate_to_nis(result)
            
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            self.logger.error(f"Failed to send to {agent_id}: {e}")
            return {
                "error": True,
                "error_message": str(e),
                "agent_id": agent_id
            }
    
    # ==================== Consciousness Pipeline Methods ====================
    
    async def run_consciousness_pipeline(
        self,
        input_data: str,
        phases: Optional[List[ConsciousnessPhase]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run the full 10-phase consciousness pipeline.
        
        Args:
            input_data: Initial input for the pipeline
            phases: Specific phases to run (default: all phases)
            context: Additional context for processing
            
        Returns:
            Combined results from all phases
        """
        if phases is None:
            phases = list(ConsciousnessPhase)
        
        results = {
            "input": input_data,
            "phases": {},
            "final_output": None,
            "metadata": {
                "start_time": time.time(),
                "phases_executed": []
            }
        }
        
        current_context = context or {}
        
        for phase in phases:
            try:
                phase_result = await self.execute_consciousness_phase(
                    phase=phase,
                    input_data=input_data if phase == ConsciousnessPhase.GENESIS else results.get("final_output", input_data),
                    context=current_context
                )
                
                results["phases"][phase.value] = phase_result
                results["metadata"]["phases_executed"].append(phase.value)
                
                # Update context with phase results
                current_context[f"{phase.value}_result"] = phase_result
                
                # Update final output
                if "output" in phase_result:
                    results["final_output"] = phase_result["output"]
                    
            except Exception as e:
                self.logger.error(f"Phase {phase.value} failed: {e}")
                results["phases"][phase.value] = {"error": str(e)}
        
        results["metadata"]["end_time"] = time.time()
        results["metadata"]["duration"] = results["metadata"]["end_time"] - results["metadata"]["start_time"]
        
        return results
    
    async def execute_consciousness_phase(
        self,
        phase: ConsciousnessPhase,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a single consciousness phase.
        
        Args:
            phase: The consciousness phase to execute
            input_data: Input for this phase
            context: Additional context
            
        Returns:
            Phase execution result
        """
        endpoint = self.V4_ENDPOINTS["consciousness"].get(phase.value)
        if not endpoint:
            raise ValueError(f"Unknown consciousness phase: {phase.value}")
        
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {
            "input": input_data,
            "context": context or {},
            "timestamp": time.time()
        }
        
        async with aiohttp.ClientSession() as session:
            headers = dict(self.session.headers)
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Phase {phase.value} failed: {error_text}")
    
    def consciousness_genesis(self, prompt: str, capability: str = "reasoning", **kwargs) -> Dict[str, Any]:
        """Execute Genesis phase - idea/agent generation.
        
        Args:
            prompt: The initial prompt for idea generation
            capability: The capability type (reasoning, vision, research, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Generated agent specification and ideas
        """
        return self._call_consciousness_endpoint("genesis", {
            "prompt": prompt,
            "capability": capability,
            **kwargs
        })
    
    def consciousness_plan(self, high_level_goal: str, goal_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Execute Plan phase - strategic planning.
        
        Args:
            high_level_goal: The high-level goal to plan for
            goal_id: Optional goal identifier (auto-generated if not provided)
            **kwargs: Additional parameters
            
        Returns:
            Strategic plan with steps
        """
        import uuid
        return self._call_consciousness_endpoint("plan", {
            "goal_id": goal_id or f"goal_{uuid.uuid4().hex[:8]}",
            "high_level_goal": high_level_goal,
            **kwargs
        })
    
    def consciousness_collective(self, proposals: List[Dict], **kwargs) -> Dict[str, Any]:
        """Execute Collective phase - multi-agent consensus.
        
        Args:
            proposals: List of proposals to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Consensus result
        """
        return self._call_consciousness_endpoint("collective", {
            "proposals": proposals,
            **kwargs
        })
    
    def consciousness_ethics(self, action: str, context: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Execute Ethics phase - ethical evaluation.
        
        Args:
            action: The action to evaluate
            context: Ethical context
            **kwargs: Additional parameters
            
        Returns:
            Ethical evaluation result
        """
        return self._call_consciousness_endpoint("ethics", {
            "action": action,
            "context": context or {},
            **kwargs
        })
    
    def consciousness_multipath_start(self, query: str, num_paths: int = 3, **kwargs) -> Dict[str, Any]:
        """Start multipath parallel reasoning.
        
        Args:
            query: The query to reason about
            num_paths: Number of parallel reasoning paths
            **kwargs: Additional parameters
            
        Returns:
            Multipath session info
        """
        return self._call_consciousness_endpoint("multipath_start", {
            "query": query,
            "num_paths": num_paths,
            **kwargs
        })
    
    def consciousness_multipath_collapse(self, session_id: str, **kwargs) -> Dict[str, Any]:
        """Collapse multipath reasoning to single result.
        
        Args:
            session_id: The multipath session ID
            **kwargs: Additional parameters
            
        Returns:
            Collapsed reasoning result
        """
        return self._call_consciousness_endpoint("multipath_collapse", {
            "session_id": session_id,
            **kwargs
        })
    
    def consciousness_evolve(self, agent_id: str, feedback: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Evolve an agent based on feedback.
        
        Args:
            agent_id: The agent to evolve
            feedback: Feedback for evolution
            **kwargs: Additional parameters
            
        Returns:
            Evolution result
        """
        return self._call_consciousness_endpoint("evolve", {
            "agent_id": agent_id,
            "feedback": feedback,
            **kwargs
        })
    
    def consciousness_embodiment_action(self, action_type: str, parameters: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute a physical embodiment action.
        
        Args:
            action_type: Type of action to execute
            parameters: Action parameters
            **kwargs: Additional parameters
            
        Returns:
            Action execution result
        """
        return self._call_consciousness_endpoint("embodiment_action", {
            "action_type": action_type,
            "parameters": parameters,
            **kwargs
        })
    
    def consciousness_marketplace_publish(self, insight: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Publish an insight to the marketplace.
        
        Args:
            insight: The insight to publish
            **kwargs: Additional parameters
            
        Returns:
            Publication result
        """
        return self._call_consciousness_endpoint("marketplace_publish", {
            "insight": insight,
            **kwargs
        })
    
    def consciousness_debug_record(self, event: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Record a debug event.
        
        Args:
            event: Event type
            data: Event data
            **kwargs: Additional parameters
            
        Returns:
            Recording result
        """
        return self._call_consciousness_endpoint("debug_record", {
            "event": event,
            "data": data,
            **kwargs
        })
    
    def consciousness_performance(self) -> Dict[str, Any]:
        """Get consciousness pipeline performance metrics.
        
        Returns:
            Performance metrics
        """
        endpoint = self.V4_ENDPOINTS["consciousness"]["performance"]
        url = f"{self.v4_config.base_url}{endpoint}"
        response = self.session.get(url, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    # ==================== Vision Methods ====================
    
    def vision_analyze(self, image_data: str, analysis_type: str = "general", **kwargs) -> Dict[str, Any]:
        """Analyze an image.
        
        Args:
            image_data: Base64 encoded image or URL
            analysis_type: Type of analysis (general, detailed, objects, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Analysis result
        """
        endpoint = self.V4_ENDPOINTS["vision"]["analyze"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {
            "image": image_data,
            "analysis_type": analysis_type,
            **kwargs
        }
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    def vision_generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate an image from a prompt.
        
        Args:
            prompt: Image generation prompt
            **kwargs: Additional parameters (size, style, etc.)
            
        Returns:
            Generated image data
        """
        endpoint = self.V4_ENDPOINTS["vision"]["generate"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {"prompt": prompt, **kwargs}
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    # ==================== Voice Methods ====================
    
    def voice_tts(self, text: str, voice: str = "default", **kwargs) -> Dict[str, Any]:
        """Convert text to speech.
        
        Args:
            text: Text to convert
            voice: Voice ID to use
            **kwargs: Additional parameters
            
        Returns:
            Audio data or URL
        """
        endpoint = self.V4_ENDPOINTS["voice"]["tts"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {"text": text, "voice": voice, **kwargs}
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    def voice_stt(self, audio_data: str, **kwargs) -> Dict[str, Any]:
        """Convert speech to text.
        
        Args:
            audio_data: Base64 encoded audio
            **kwargs: Additional parameters
            
        Returns:
            Transcription result
        """
        endpoint = self.V4_ENDPOINTS["voice"]["stt"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {"audio": audio_data, **kwargs}
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    # ==================== Research Methods ====================
    
    def research_search(self, query: str, sources: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Search for research information.
        
        Args:
            query: Search query
            sources: Sources to search (web, arxiv, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Search results
        """
        endpoint = self.V4_ENDPOINTS["research"]["search"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {
            "query": query,
            "sources": sources or ["web"],
            **kwargs
        }
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    def research_arxiv(self, query: str, max_results: int = 10, **kwargs) -> Dict[str, Any]:
        """Search ArXiv for papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            **kwargs: Additional parameters
            
        Returns:
            ArXiv search results
        """
        endpoint = self.V4_ENDPOINTS["research"]["arxiv"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {
            "query": query,
            "max_results": max_results,
            **kwargs
        }
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    # ==================== BitNet (Local AI) Methods ====================
    
    def bitnet_status(self) -> Dict[str, Any]:
        """Get BitNet local AI status.
        
        Returns:
            BitNet status and available models
        """
        endpoint = self.V4_ENDPOINTS["bitnet"]["status"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        response = self.session.get(url, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    def bitnet_generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate text using BitNet local AI.
        
        Args:
            prompt: Generation prompt
            model: Model to use (optional)
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        endpoint = self.V4_ENDPOINTS["bitnet"]["generate"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {"prompt": prompt, **kwargs}
        if model:
            payload["model"] = model
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    # ==================== Robotics Methods ====================
    
    def robotics_forward_kinematics(
        self,
        joint_angles: List[float],
        robot_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Calculate forward kinematics.
        
        Args:
            joint_angles: List of joint angles in radians
            robot_config: Robot configuration (DH parameters, etc.)
            
        Returns:
            End effector pose
        """
        endpoint = self.V4_ENDPOINTS["robotics"]["forward_kinematics"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {
            "joint_angles": joint_angles,
            "robot_config": robot_config or {}
        }
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    def robotics_inverse_kinematics(
        self,
        target_pose: Dict[str, Any],
        robot_config: Optional[Dict] = None,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Calculate inverse kinematics.
        
        Args:
            target_pose: Target end effector pose
            robot_config: Robot configuration
            constraints: Joint constraints
            
        Returns:
            Joint angles to achieve target pose
        """
        endpoint = self.V4_ENDPOINTS["robotics"]["inverse_kinematics"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {
            "target_pose": target_pose,
            "robot_config": robot_config or {},
            "constraints": constraints or {}
        }
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    def robotics_plan_trajectory(
        self,
        start_pose: Dict[str, Any],
        end_pose: Dict[str, Any],
        waypoints: Optional[List[Dict]] = None,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Plan a trajectory between poses.
        
        Args:
            start_pose: Starting pose
            end_pose: Target pose
            waypoints: Optional intermediate waypoints
            constraints: Trajectory constraints
            
        Returns:
            Planned trajectory
        """
        endpoint = self.V4_ENDPOINTS["robotics"]["plan_trajectory"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {
            "start_pose": start_pose,
            "end_pose": end_pose,
            "waypoints": waypoints or [],
            "constraints": constraints or {}
        }
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    def robotics_capabilities(self) -> Dict[str, Any]:
        """Get robotics system capabilities.
        
        Returns:
            Available robotics capabilities
        """
        endpoint = self.V4_ENDPOINTS["robotics"]["capabilities"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        response = self.session.get(url, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    # ==================== Physics Methods ====================
    
    def physics_validate(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Validate physics constraints.
        
        Args:
            constraints: Physics constraints to validate
            
        Returns:
            Validation result
        """
        endpoint = self.V4_ENDPOINTS["physics"]["validate"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        response = self.session.post(url, json=constraints, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    def physics_solve_heat_equation(
        self,
        parameters: Dict[str, Any],
        boundary_conditions: Dict[str, Any],
        initial_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve heat equation using PINN.
        
        Args:
            parameters: Equation parameters
            boundary_conditions: Boundary conditions
            initial_conditions: Initial conditions
            
        Returns:
            Solution
        """
        endpoint = self.V4_ENDPOINTS["physics"]["heat_equation"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {
            "parameters": parameters,
            "boundary_conditions": boundary_conditions,
            "initial_conditions": initial_conditions
        }
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    def physics_constants(self) -> Dict[str, Any]:
        """Get physical constants.
        
        Returns:
            Dictionary of physical constants
        """
        endpoint = self.V4_ENDPOINTS["physics"]["constants"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        response = self.session.get(url, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    # ==================== Authentication Methods ====================
    
    def auth_login(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate with NIS Protocol v4.0.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Authentication result with JWT token
        """
        endpoint = self.V4_ENDPOINTS["auth"]["login"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {
            "username": username,
            "password": password
        }
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        result = response.json()
        
        # Update JWT token if provided
        if "token" in result:
            self.v4_config.jwt_token = result["token"]
            self._setup_auth_headers()
        
        return result
    
    def auth_verify(self) -> Dict[str, Any]:
        """Verify current authentication.
        
        Returns:
            Verification result
        """
        endpoint = self.V4_ENDPOINTS["auth"]["verify"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        response = self.session.get(url, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    def auth_create_api_key(self, name: str, permissions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new API key.
        
        Args:
            name: Name for the API key
            permissions: List of permissions
            
        Returns:
            Created API key
        """
        endpoint = self.V4_ENDPOINTS["auth"]["api_keys"]
        url = f"{self.v4_config.base_url}{endpoint}"
        
        payload = {
            "name": name,
            "permissions": permissions or ["read", "write"]
        }
        
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    # ==================== WebSocket Methods ====================
    
    async def connect_websocket(self, endpoint: str = "/ws/agentic") -> None:
        """Connect to WebSocket for real-time communication.
        
        Args:
            endpoint: WebSocket endpoint
        """
        url = f"{self.v4_config.websocket_url}{endpoint}"
        
        self._ws_connection = await aiohttp.ClientSession().ws_connect(url)
        self.logger.info(f"WebSocket connected to {url}")
    
    async def send_websocket_message(self, message: Dict[str, Any]) -> None:
        """Send message over WebSocket.
        
        Args:
            message: Message to send
        """
        if self._ws_connection is None:
            raise ConnectionError("WebSocket not connected")
        
        await self._ws_connection.send_json(message)
    
    async def receive_websocket_message(self) -> Dict[str, Any]:
        """Receive message from WebSocket.
        
        Returns:
            Received message
        """
        if self._ws_connection is None:
            raise ConnectionError("WebSocket not connected")
        
        msg = await self._ws_connection.receive()
        if msg.type == aiohttp.WSMsgType.TEXT:
            return json.loads(msg.data)
        return {"type": "binary", "data": msg.data}
    
    async def close_websocket(self) -> None:
        """Close WebSocket connection."""
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None
    
    # ==================== Helper Methods ====================
    
    def _detect_v4_message_type(self, message: Dict[str, Any]) -> str:
        """Detect the type of v4 message based on content."""
        endpoint = message.get("_endpoint", "")
        
        if "/consciousness/" in endpoint:
            return "consciousness"
        elif "/robotics/" in endpoint:
            return "robotics"
        elif "/physics/" in endpoint:
            return "physics"
        elif "/auth/" in endpoint:
            return "auth"
        
        # Check content-based detection
        if "phase" in message or "consciousness" in message:
            return "consciousness"
        if "joint_angles" in message or "trajectory" in message:
            return "robotics"
        if "equation" in message or "pinn" in message.get("solver", ""):
            return "physics"
        
        return "generic"
    
    def _translate_consciousness_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Translate consciousness pipeline response."""
        return {
            "phase": response.get("phase", "unknown"),
            "output": response.get("output", response.get("result", {})),
            "confidence": response.get("confidence", 1.0),
            "reasoning": response.get("reasoning", []),
            "metadata": response.get("metadata", {})
        }
    
    def _translate_robotics_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Translate robotics response."""
        return {
            "pose": response.get("pose", response.get("end_effector_pose", {})),
            "joint_angles": response.get("joint_angles", []),
            "trajectory": response.get("trajectory", []),
            "success": response.get("success", True),
            "metadata": response.get("metadata", {})
        }
    
    def _translate_physics_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Translate physics response."""
        return {
            "solution": response.get("solution", {}),
            "validation": response.get("validation", {}),
            "constraints_satisfied": response.get("constraints_satisfied", True),
            "metadata": response.get("metadata", {})
        }
    
    def _translate_auth_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Translate auth response."""
        return {
            "authenticated": response.get("authenticated", False),
            "token": response.get("token"),
            "user": response.get("user", {}),
            "permissions": response.get("permissions", [])
        }
    
    def _call_consciousness_endpoint(self, phase: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call a consciousness endpoint synchronously."""
        endpoint = self.V4_ENDPOINTS["consciousness"].get(phase)
        if not endpoint:
            raise ValueError(f"Unknown consciousness phase: {phase}")
        
        url = f"{self.v4_config.base_url}{endpoint}"
        response = self.session.post(url, json=payload, timeout=self.v4_config.timeout)
        response.raise_for_status()
        return response.json()
    
    def _resolve_endpoint(self, agent_id: str) -> str:
        """Resolve agent ID to API endpoint."""
        # Check if it's a direct endpoint
        if agent_id.startswith("/"):
            return agent_id
        
        # Check consciousness phases
        if agent_id in self.V4_ENDPOINTS["consciousness"]:
            return self.V4_ENDPOINTS["consciousness"][agent_id]
        
        # Check robotics
        if agent_id in self.V4_ENDPOINTS["robotics"]:
            return self.V4_ENDPOINTS["robotics"][agent_id]
        
        # Check physics
        if agent_id in self.V4_ENDPOINTS["physics"]:
            return self.V4_ENDPOINTS["physics"][agent_id]
        
        # Default to process endpoint
        return f"/agents/{agent_id}/process"
    
    def get_status(self) -> Dict[str, Any]:
        """Get adapter status including v4-specific information."""
        base_status = {
            "connection_status": self.connection_status,
            "error_count": self.error_count,
            "message_count": len(self.message_history),
            "last_heartbeat": self.last_heartbeat,
            "protocol_name": "NIS_v4"
        }
        
        # Add v4-specific status
        base_status["v4_config"] = {
            "base_url": self.v4_config.base_url,
            "websocket_enabled": self.v4_config.enable_websocket,
            "robotics_enabled": self.v4_config.enable_robotics,
            "consciousness_enabled": self.v4_config.enable_consciousness,
            "physics_enabled": self.v4_config.enable_physics,
            "bitnet_enabled": self.v4_config.enable_bitnet,
            "flutter_mode": self.v4_config.flutter_client_mode
        }
        
        base_status["websocket_connected"] = self._ws_connection is not None
        
        return base_status


# Convenience function for quick setup
def create_nis_v4_adapter(
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    jwt_token: Optional[str] = None,
    **kwargs
) -> NISv4Adapter:
    """Create a configured NIS v4.0 adapter.
    
    Args:
        base_url: NIS Protocol v4.0 server URL
        api_key: API key for authentication
        jwt_token: JWT token for authentication
        **kwargs: Additional configuration options
        
    Returns:
        Configured NISv4Adapter instance
    """
    config = NISv4Config(
        base_url=base_url,
        api_key=api_key,
        jwt_token=jwt_token,
        **kwargs
    )
    return NISv4Adapter({"v4_config": config})
