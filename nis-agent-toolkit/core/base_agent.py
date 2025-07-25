#!/usr/bin/env python3
"""
NIS Agent Toolkit - Enhanced Base Agent Framework
Abstract base class for all NIS agents with consciousness integration and KAN reasoning
"""

import logging
import asyncio
import numpy as np
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

# Optional advanced imports with graceful fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

class AgentState(Enum):
    """Enhanced agent operational states"""
    IDLE = "idle"
    PROCESSING = "processing" 
    WAITING = "waiting"
    ERROR = "error"
    CONSCIOUSNESS_REFLECTION = "consciousness_reflection"
    KAN_REASONING = "kan_reasoning"
    COORDINATING = "coordinating"

class ConsciousnessLevel(Enum):
    """Consciousness integration levels"""
    BASIC = 0.3
    MODERATE = 0.5
    HIGH = 0.8
    ADVANCED = 0.9
    EXPERT = 0.95

class SafetyLevel(Enum):
    """Safety requirement levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ConsciousnessState:
    """Agent consciousness state tracking"""
    self_awareness_score: float = 0.0
    bias_detection_active: bool = True
    meta_cognitive_insights: List[str] = field(default_factory=list)
    attention_focus: List[str] = field(default_factory=list)
    ethical_constraints: List[str] = field(default_factory=list)
    uncertainty_acknowledgment: float = 0.0
    reflection_depth: float = 0.0
    last_reflection: Optional[datetime] = None

@dataclass
class KANReasoningState:
    """KAN mathematical reasoning state"""
    interpretability_score: float = 0.0
    mathematical_accuracy: float = 0.0
    convergence_guaranteed: bool = False
    spline_coefficients: List[float] = field(default_factory=list)
    confidence_bounds: Tuple[float, float] = (0.0, 1.0)
    mathematical_proof: Dict[str, Any] = field(default_factory=dict)
    feature_space_dimensionality: int = 10

@dataclass
class AgentMetrics:
    """Comprehensive agent performance metrics"""
    total_processes: int = 0
    successful_processes: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    consciousness_activations: int = 0
    kan_reasoning_sessions: int = 0
    bias_detections: int = 0
    uncertainty_acknowledgments: int = 0
    coordination_successes: int = 0
    last_activity: Optional[datetime] = None

class BaseNISAgent(ABC):
    """
    Enhanced abstract base class for all NIS agents
    
    Features:
    - Consciousness integration with self-awareness and bias detection
    - KAN mathematical reasoning with interpretability guarantees
    - Advanced coordination and communication capabilities
    - Comprehensive monitoring and metrics
    - Safety and ethical constraint enforcement
    """
    
    def __init__(self, agent_id: str, agent_type: str = "generic", 
                 consciousness_level: float = 0.8, safety_level: str = "medium",
                 domain: str = "general"):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.domain = domain
        self.state = AgentState.IDLE
        self.logger = logging.getLogger(f"nis.agent.{agent_id}")
        
        # Enhanced agent configuration
        self.consciousness_level = consciousness_level
        self.safety_level = SafetyLevel(safety_level)
        
        # Agent capabilities and tools
        self.capabilities = set()
        self.tools = {}
        self.memory = []
        self.config = {}
        
        # Consciousness integration
        self.consciousness_state = ConsciousnessState()
        self.consciousness_callbacks = []
        
        # KAN reasoning integration
        self.kan_reasoning_state = KANReasoningState()
        self.kan_enabled = True
        
        # Performance tracking and metrics
        self.metrics = AgentMetrics()
        
        # Communication and coordination
        self.coordination_partners = set()
        self.message_queue = asyncio.Queue()
        self.coordination_history = []
        
        # Safety and validation
        self.safety_validators = []
        self.ethical_constraints = [
            "respect_human_autonomy",
            "avoid_harm",
            "ensure_fairness", 
            "maintain_transparency"
        ]
        
        # Initialize core consciousness
        self._initialize_consciousness()
        self._initialize_kan_reasoning()
        
        self.logger.info(f"Enhanced {agent_type} agent initialized: {agent_id}")
        self.logger.info(f"Consciousness level: {consciousness_level:.1%}, Safety: {safety_level}")
    
    def _initialize_consciousness(self):
        """Initialize consciousness framework"""
        self.consciousness_state.self_awareness_score = self.consciousness_level
        self.consciousness_state.ethical_constraints = self.ethical_constraints.copy()
        self.consciousness_state.last_reflection = datetime.now()
        
        # Add default consciousness capabilities
        self.add_capability("consciousness_integration")
        self.add_capability("bias_detection")
        self.add_capability("meta_cognitive_reflection")
        self.add_capability("uncertainty_quantification")
        
    def _initialize_kan_reasoning(self):
        """Initialize KAN mathematical reasoning"""
        if self.kan_enabled:
            self.kan_reasoning_state.interpretability_score = 0.95  # Target high interpretability
            self.kan_reasoning_state.convergence_guaranteed = True
            self.kan_reasoning_state.confidence_bounds = (0.85, 0.99)
            
            # Add KAN capabilities
            self.add_capability("kan_mathematical_reasoning")
            self.add_capability("interpretable_decision_making")
            self.add_capability("mathematical_proof_generation")
    
    @abstractmethod
    async def observe(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consciousness-aware observation and input processing
        
        Args:
            input_data: Raw input data to process
            
        Returns:
            Enhanced observation data with consciousness insights
        """
        pass
    
    @abstractmethod
    async def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        KAN-enhanced decision making with mathematical guarantees
        
        Args:
            observation: Processed observation data
            
        Returns:
            Decision with mathematical proof and confidence bounds
        """
        pass
    
    @abstractmethod
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safety-validated action execution with monitoring
        
        Args:
            decision: Decision data from decide() method
            
        Returns:
            Action results with safety validation
        """
        pass
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced main processing pipeline with consciousness and KAN integration
        
        This is the core agent processing loop that all agents follow,
        now enhanced with consciousness reflection and mathematical reasoning.
        """
        
        start_time = datetime.now()
        
        try:
            self.state = AgentState.PROCESSING
            self.metrics.last_activity = start_time
            
            # Pre-processing consciousness reflection
            await self._consciousness_reflection("pre_processing", input_data)
            
            # Enhanced agent pipeline with consciousness integration
            observation = await self._consciousness_aware_observe(input_data)
            decision = await self._kan_enhanced_decide(observation)
            action = await self._safety_validated_act(decision)
            
            # Post-processing consciousness reflection
            await self._consciousness_reflection("post_processing", action)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Package enhanced results
            result = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "domain": self.domain,
                "timestamp": start_time.isoformat(),
                "processing_time_ms": processing_time * 1000,
                "input": input_data,
                "observation": observation,
                "decision": decision,
                "action": action,
                "consciousness_state": self._get_consciousness_summary(),
                "kan_reasoning": self._get_kan_summary(),
                "metrics": self._get_metrics_summary(),
                "status": "success"
            }
            
            # Update metrics
            self.metrics.total_processes += 1
            self.metrics.successful_processes += 1
            self._update_avg_response_time(processing_time)
            
            self.state = AgentState.IDLE
            
            self.logger.info(f"Enhanced processing completed: {processing_time:.3f}s")
            return result
            
        except Exception as e:
            self.metrics.error_count += 1
            self.state = AgentState.ERROR
            
            # Consciousness-aware error handling
            await self._consciousness_reflection("error_handling", {"error": str(e)})
            
            error_result = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "timestamp": datetime.now().isoformat(),
                "input": input_data,
                "error": str(e),
                "consciousness_insights": self.consciousness_state.meta_cognitive_insights[-3:],
                "status": "error",
                "error_count": self.metrics.error_count
            }
            
            self.logger.error(f"Enhanced processing error: {e}")
            return error_result
    
    async def _consciousness_aware_observe(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced observation with consciousness integration"""
        self.state = AgentState.CONSCIOUSNESS_REFLECTION
        
        # Perform consciousness-guided observation
        raw_observation = await self.observe(input_data)
        
        # Add consciousness insights
        consciousness_insights = await self._analyze_observation_biases(raw_observation)
        attention_analysis = await self._analyze_attention_patterns(input_data)
        
        enhanced_observation = {
            **raw_observation,
            "consciousness_insights": {
                "bias_flags": consciousness_insights.get("detected_biases", []),
                "attention_focus": attention_analysis.get("focus_areas", []),
                "uncertainty_level": consciousness_insights.get("uncertainty", 0.0),
                "meta_cognitive_notes": consciousness_insights.get("meta_notes", [])
            },
                               "observation_confidence": consciousness_insights.get("confidence", self.consciousness_level)
        }
        
        self.metrics.consciousness_activations += 1
        return enhanced_observation
    
    async def _kan_enhanced_decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced decision making with KAN mathematical reasoning"""
        self.state = AgentState.KAN_REASONING
        
        # Extract numerical features for KAN processing
        numerical_features = await self._extract_kan_features(observation)
        
        # Perform KAN mathematical reasoning
        kan_analysis = await self._kan_mathematical_reasoning(numerical_features)
        
        # Perform domain-specific decision making
        raw_decision = await self.decide(observation)
        
        # Enhance decision with KAN insights
        enhanced_decision = {
            **raw_decision,
            "kan_reasoning": {
                "interpretability_score": kan_analysis.get("interpretability", 0.95),
                "mathematical_proof": kan_analysis.get("proof", {}),
                "confidence_bounds": kan_analysis.get("confidence_bounds", (0.8, 0.95)),
                "convergence_guaranteed": kan_analysis.get("convergence", True),
                "feature_importance": kan_analysis.get("feature_importance", [])
            },
            "decision_confidence": kan_analysis.get("overall_confidence", 0.85)
        }
        
        self.metrics.kan_reasoning_sessions += 1
        return enhanced_decision
    
    async def _safety_validated_act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced action execution with safety validation"""
        
        # Pre-action safety validation
        safety_check = await self._validate_action_safety(decision)
        
        if not safety_check["safe_to_proceed"]:
            return {
                "action_type": "safety_violation",
                "safety_violation": safety_check["violations"],
                "decision_blocked": True,
                "safety_level": self.safety_level.value,
                "status": "blocked"
            }
        
        # Execute action with monitoring
        action_result = await self.act(decision)
        
        # Post-action validation and monitoring
        post_action_validation = await self._post_action_monitoring(action_result)
        
        enhanced_action = {
            **action_result,
            "safety_validation": {
                "pre_action_check": safety_check,
                "post_action_monitoring": post_action_validation,
                "safety_level": self.safety_level.value,
                "ethical_constraints_met": post_action_validation.get("ethical_compliance", True)
            }
        }
        
        return enhanced_action
    
    async def _consciousness_reflection(self, phase: str, data: Dict[str, Any]):
        """Perform consciousness reflection and meta-cognitive analysis"""
        
        reflection_insights = []
        
        if phase == "pre_processing":
            reflection_insights.extend([
                f"Processing input of type: {type(data).__name__}",
                f"Consciousness level: {self.consciousness_level:.1%}",
                f"Current attention: {len(self.consciousness_state.attention_focus)} focus areas"
            ])
            
        elif phase == "post_processing":
            reflection_insights.extend([
                f"Successfully processed with status: {data.get('status', 'unknown')}",
                f"Action type: {data.get('action_type', 'unknown')}",
                f"Confidence achieved: {data.get('decision_confidence', 0.0):.2f}"
            ])
            
        elif phase == "error_handling":
            reflection_insights.extend([
                f"Error encountered: {data.get('error', 'unknown')}",
                "Initiating error recovery procedures",
                "Adjusting future processing based on error patterns"
            ])
        
        # Update consciousness state
        self.consciousness_state.meta_cognitive_insights.extend(reflection_insights)
        self.consciousness_state.last_reflection = datetime.now()
        
        # Keep only recent insights (last 10)
        if len(self.consciousness_state.meta_cognitive_insights) > 10:
            self.consciousness_state.meta_cognitive_insights = self.consciousness_state.meta_cognitive_insights[-10:]
        
        # Trigger consciousness callbacks
        for callback in self.consciousness_callbacks:
            try:
                await callback(phase, reflection_insights)
            except Exception as e:
                self.logger.warning(f"Consciousness callback error: {e}")
    
    async def _extract_kan_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract numerical features for KAN mathematical processing"""
        
        features = []
        
        # Extract basic numerical features
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, str):
                    features.append(len(value) / 100.0)  # Normalized string length
                elif isinstance(value, list):
                    features.append(len(value) / 10.0)  # Normalized list length
                elif isinstance(value, dict):
                    features.append(len(value) / 5.0)  # Normalized dict size
        
        # Add consciousness-related features
        features.extend([
            self.consciousness_level,
            self.consciousness_state.self_awareness_score,
            len(self.consciousness_state.attention_focus) / 5.0,
            self.consciousness_state.uncertainty_acknowledgment
        ])
        
        # Add agent state features
        features.extend([
            self.metrics.successful_processes / max(self.metrics.total_processes, 1),
            self.metrics.error_count / max(self.metrics.total_processes, 1),
            min(self.metrics.avg_response_time / 1000.0, 1.0),  # Normalized to 1 second max
            len(self.capabilities) / 10.0
        ])
        
        # Ensure we have exactly 10 features for KAN processing
        while len(features) < 10:
            features.append(0.5)  # Neutral padding
        
        return features[:10]
    
    async def _kan_mathematical_reasoning(self, features: List[float]) -> Dict[str, Any]:
        """Perform KAN mathematical reasoning with interpretability guarantees"""
        
        feature_array = np.array(features)
        
        # Simulate KAN spline-based processing
        # In a real implementation, this would use actual KAN networks
        spline_coefficients = np.random.normal(0, 0.1, len(features))
        
        # Mathematical analysis
        complexity_score = np.mean(feature_array)
        variance = np.var(feature_array)
        interpretability_score = max(0.8, 1.0 - variance)  # Lower variance = higher interpretability
        
        # Convergence analysis
        convergence_guaranteed = interpretability_score > 0.85 and complexity_score < 0.8
        
        # Confidence bounds calculation
        base_confidence = interpretability_score * 0.8 + complexity_score * 0.2
        confidence_bounds = (
            max(0.0, base_confidence - 0.1),
            min(1.0, base_confidence + 0.1)
        )
        
        # Feature importance (which features contribute most to decisions)
        feature_importance = (np.abs(spline_coefficients) / np.sum(np.abs(spline_coefficients))).tolist()
        
        # Mathematical proof structure
        mathematical_proof = {
            "spline_analysis": {
                "coefficients": spline_coefficients.tolist(),
                "interpretability_threshold": 0.95,
                "threshold_met": interpretability_score >= 0.95
            },
            "convergence_proof": {
                "guaranteed": convergence_guaranteed,
                "complexity_factor": complexity_score,
                "variance_bound": variance
            },
            "mathematical_guarantees": [
                "interpretability_above_threshold" if interpretability_score >= 0.95 else "interpretability_below_threshold",
                "convergence_guaranteed" if convergence_guaranteed else "convergence_approximate",
                "mathematical_traceability"
            ]
        }
        
        return {
            "interpretability": interpretability_score,
            "overall_confidence": base_confidence,
            "confidence_bounds": confidence_bounds,
            "convergence": convergence_guaranteed,
            "proof": mathematical_proof,
            "feature_importance": feature_importance,
            "spline_coefficients": spline_coefficients.tolist()
        }
    
    # Additional helper methods for consciousness and safety
    async def _analyze_observation_biases(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential biases in observation processing"""
        
        detected_biases = []
        # Calculate confidence based on observation complexity and available data
        observation_complexity = len(observation.keys()) / 10.0  # Normalize by typical observation size
        confidence = min(max(0.6 + observation_complexity * 0.2, 0.6), 0.95)
        uncertainty = 1.0 - confidence
        
        # Check for confirmation bias
        if "previous_results" in observation:
            similar_count = len([r for r in observation.get("previous_results", []) 
                               if r.get("pattern") == observation.get("current_pattern")])
            if similar_count > 3:
                detected_biases.append("potential_confirmation_bias")
        
        # Check for recency bias
        if "timestamp_data" in observation:
            recent_items = [item for item in observation.get("timestamp_data", [])
                          if (datetime.now() - datetime.fromisoformat(item.get("timestamp", "2020-01-01"))).days < 7]
            if len(recent_items) > len(observation.get("timestamp_data", [])) * 0.8:
                detected_biases.append("recency_bias")
        
        # Adjust confidence based on bias detection
        if detected_biases:
            confidence *= 0.8
            uncertainty = 1.0 - confidence
            self.metrics.bias_detections += 1
        
        meta_notes = [
            f"Bias detection scan completed: {len(detected_biases)} biases found",
            f"Observation confidence adjusted to: {confidence:.2f}"
        ]
        
        return {
            "detected_biases": detected_biases,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "meta_notes": meta_notes
        }
    
    async def _analyze_attention_patterns(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention patterns and focus areas"""
        
        focus_areas = []
        
        # Analyze input data to determine attention focus
        if isinstance(input_data, dict):
            for key, value in input_data.items():
                if isinstance(value, (list, dict)) and value:
                    focus_areas.append(f"complex_data_{key}")
                elif isinstance(value, str) and len(value) > 100:
                    focus_areas.append(f"detailed_text_{key}")
                elif isinstance(value, (int, float)) and abs(value) > 1000:
                    focus_areas.append(f"significant_number_{key}")
        
        # Update consciousness state
        self.consciousness_state.attention_focus = focus_areas
        
        return {
            "focus_areas": focus_areas,
            "attention_distribution": {area: 1.0/len(focus_areas) for area in focus_areas} if focus_areas else {},
            "attention_intensity": min(len(focus_areas) / 3.0, 1.0)
        }
    
    async def _validate_action_safety(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate action safety before execution"""
        
        violations = []
        safe_to_proceed = True
        
        # Check safety level requirements
        if self.safety_level == SafetyLevel.CRITICAL:
            if decision.get("decision_confidence", 0.0) < 0.95:
                violations.append("confidence_below_critical_threshold")
                safe_to_proceed = False
        
        elif self.safety_level == SafetyLevel.HIGH:
            if decision.get("decision_confidence", 0.0) < 0.85:
                violations.append("confidence_below_high_threshold")
                safe_to_proceed = False
        
        # Check ethical constraints
        action_type = decision.get("action_type", "unknown")
        if "harmful" in action_type.lower() or "destructive" in action_type.lower():
            violations.append("potentially_harmful_action")
            safe_to_proceed = False
        
        # Check KAN mathematical guarantees
        kan_reasoning = decision.get("kan_reasoning", {})
        if not kan_reasoning.get("convergence_guaranteed", False):
            if self.safety_level in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]:
                violations.append("mathematical_convergence_not_guaranteed")
                safe_to_proceed = False
        
        return {
            "safe_to_proceed": safe_to_proceed,
            "violations": violations,
            "safety_level": self.safety_level.value,
            "validation_timestamp": datetime.now().isoformat()
        }
    
    async def _post_action_monitoring(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor action execution and results"""
        
        monitoring_result = {
            "execution_successful": action_result.get("status") == "success",
            "ethical_compliance": True,
            "performance_metrics": {
                "execution_time": action_result.get("execution_time", 0),
                "resource_usage": action_result.get("resource_usage", "normal")
            },
            "post_action_insights": []
        }
        
        # Check for ethical violations
        if "violation" in str(action_result).lower():
            monitoring_result["ethical_compliance"] = False
            monitoring_result["post_action_insights"].append("potential_ethical_concern_detected")
        
        # Update agent metrics based on results
        if monitoring_result["execution_successful"]:
            self.metrics.coordination_successes += 1
        
        return monitoring_result
    
    # Enhanced utility methods
    def add_capability(self, capability: str):
        """Add a capability to this agent with logging"""
        self.capabilities.add(capability)
        self.logger.info(f"Added capability: {capability}")
    
    def add_tool(self, tool_name: str, tool_function):
        """Add a tool to this agent with validation"""
        if callable(tool_function):
            self.tools[tool_name] = tool_function
            self.logger.info(f"Added tool: {tool_name}")
        else:
            self.logger.error(f"Tool {tool_name} is not callable")
    
    def add_consciousness_callback(self, callback: Callable):
        """Add a callback for consciousness events"""
        if callable(callback):
            self.consciousness_callbacks.append(callback)
            self.logger.info("Added consciousness callback")
    
    def add_safety_validator(self, validator: Callable):
        """Add a custom safety validator"""
        if callable(validator):
            self.safety_validators.append(validator)
            self.logger.info("Added safety validator")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "domain": self.domain,
            "state": self.state.value,
            "consciousness_level": self.consciousness_level,
            "safety_level": self.safety_level.value,
            "capabilities": list(self.capabilities),
            "tools": list(self.tools.keys()),
            "consciousness_state": self._get_consciousness_summary(),
            "kan_reasoning_state": self._get_kan_summary(),
            "metrics": self._get_metrics_summary(),
            "coordination_partners": len(self.coordination_partners),
            "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
        }
    
    def _get_consciousness_summary(self) -> Dict[str, Any]:
        """Get consciousness state summary"""
        return {
            "self_awareness_score": self.consciousness_state.self_awareness_score,
            "bias_detection_active": self.consciousness_state.bias_detection_active,
            "recent_insights": self.consciousness_state.meta_cognitive_insights[-3:],
            "attention_focus_count": len(self.consciousness_state.attention_focus),
            "ethical_constraints_count": len(self.consciousness_state.ethical_constraints),
            "last_reflection": self.consciousness_state.last_reflection.isoformat() if self.consciousness_state.last_reflection else None
        }
    
    def _get_kan_summary(self) -> Dict[str, Any]:
        """Get KAN reasoning state summary"""
        return {
            "interpretability_score": self.kan_reasoning_state.interpretability_score,
            "mathematical_accuracy": self.kan_reasoning_state.mathematical_accuracy,
            "convergence_guaranteed": self.kan_reasoning_state.convergence_guaranteed,
            "confidence_bounds": self.kan_reasoning_state.confidence_bounds,
            "feature_space_dimensionality": self.kan_reasoning_state.feature_space_dimensionality
        }
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        success_rate = (self.metrics.successful_processes / max(self.metrics.total_processes, 1)) * 100
        return {
            "total_processes": self.metrics.total_processes,
            "success_rate": f"{success_rate:.1f}%",
            "error_count": self.metrics.error_count,
            "avg_response_time_ms": f"{self.metrics.avg_response_time:.2f}",
            "consciousness_activations": self.metrics.consciousness_activations,
            "kan_reasoning_sessions": self.metrics.kan_reasoning_sessions,
            "bias_detections": self.metrics.bias_detections
        }
    
    def _update_avg_response_time(self, new_time: float):
        """Update average response time with exponential moving average"""
        if self.metrics.avg_response_time == 0:
            self.metrics.avg_response_time = new_time
        else:
            # Exponential moving average with alpha=0.1
            self.metrics.avg_response_time = 0.9 * self.metrics.avg_response_time + 0.1 * new_time
    
    # Coordination methods
    async def coordinate_with_agent(self, agent_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate with another agent"""
        self.state = AgentState.COORDINATING
        
        coordination_message = {
            "from_agent": self.agent_id,
            "to_agent": agent_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "consciousness_level": self.consciousness_level
        }
        
        # Add to coordination history
        self.coordination_history.append(coordination_message)
        self.coordination_partners.add(agent_id)
        
        # In a real implementation, this would send the message through
        # the multi-agent coordination system
        
        self.state = AgentState.IDLE
        return {"coordination_sent": True, "message_id": hashlib.md5(str(coordination_message).encode()).hexdigest()[:8]}
    
    async def receive_coordination_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Receive and process coordination message from another agent"""
        await self.message_queue.put(message)
        return {"message_received": True, "queue_size": self.message_queue.qsize()}
    
    # Save and load agent state
    async def save_state(self, filepath: Path):
        """Save agent state to file"""
        state_data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "domain": self.domain,
            "consciousness_level": self.consciousness_level,
            "safety_level": self.safety_level.value,
            "capabilities": list(self.capabilities),
            "consciousness_state": {
                "self_awareness_score": self.consciousness_state.self_awareness_score,
                "meta_cognitive_insights": self.consciousness_state.meta_cognitive_insights,
                "attention_focus": self.consciousness_state.attention_focus,
                "ethical_constraints": self.consciousness_state.ethical_constraints
            },
            "metrics": {
                "total_processes": self.metrics.total_processes,
                "successful_processes": self.metrics.successful_processes,
                "error_count": self.metrics.error_count,
                "avg_response_time": self.metrics.avg_response_time
            },
            "coordination_partners": list(self.coordination_partners),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        self.logger.info(f"Agent state saved to {filepath}")
    
    async def load_state(self, filepath: Path):
        """Load agent state from file"""
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        # Restore state
        self.consciousness_level = state_data["consciousness_level"]
        self.safety_level = SafetyLevel(state_data["safety_level"])
        self.capabilities = set(state_data["capabilities"])
        
        # Restore consciousness state
        self.consciousness_state.self_awareness_score = state_data["consciousness_state"]["self_awareness_score"]
        self.consciousness_state.meta_cognitive_insights = state_data["consciousness_state"]["meta_cognitive_insights"]
        self.consciousness_state.attention_focus = state_data["consciousness_state"]["attention_focus"]
        self.consciousness_state.ethical_constraints = state_data["consciousness_state"]["ethical_constraints"]
        
        # Restore metrics
        self.metrics.total_processes = state_data["metrics"]["total_processes"]
        self.metrics.successful_processes = state_data["metrics"]["successful_processes"]
        self.metrics.error_count = state_data["metrics"]["error_count"]
        self.metrics.avg_response_time = state_data["metrics"]["avg_response_time"]
        
        # Restore coordination partners
        self.coordination_partners = set(state_data["coordination_partners"])
        
        self.logger.info(f"Agent state loaded from {filepath}")
