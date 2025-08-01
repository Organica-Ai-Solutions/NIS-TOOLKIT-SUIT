#!/usr/bin/env python3
"""
NIS Protocol v3.0 Consciousness Interface
Practical interface for integrating with meta-cognitive processing capabilities
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ConsciousnessState:
    """Represents the current consciousness state"""
    self_awareness_score: float
    cognitive_load: float
    bias_flags: List[str]
    meta_cognitive_insights: List[str]
    introspection_depth: float
    emotional_state: Dict[str, float]
    attention_focus: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "self_awareness_score": self.self_awareness_score,
            "cognitive_load": self.cognitive_load,
            "bias_flags": self.bias_flags,
            "meta_cognitive_insights": self.meta_cognitive_insights,
            "introspection_depth": self.introspection_depth,
            "emotional_state": self.emotional_state,
            "attention_focus": self.attention_focus
        }

@dataclass
class ConsciousnessConfig:
    """Configuration for consciousness processing"""
    meta_cognitive_processing: bool = True
    bias_detection: bool = True
    self_reflection_interval: int = 300  # seconds
    introspection_depth: float = 0.8
    emotional_awareness: bool = True
    attention_tracking: bool = True
    consciousness_threshold: float = 0.7
    
    def validate(self) -> Dict[str, Any]:
        """Validate consciousness configuration"""
        errors = []
        
        if not self.meta_cognitive_processing:
            errors.append("Meta-cognitive processing is required for consciousness")
        
        if not self.bias_detection:
            errors.append("Bias detection is essential for conscious processing")
        
        if self.introspection_depth < 0.5:
            errors.append("Introspection depth must be at least 0.5")
        
        if self.consciousness_threshold < 0.6:
            errors.append("Consciousness threshold must be at least 0.6")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "consciousness_score": self.introspection_depth * 0.6 + self.consciousness_threshold * 0.4
        }

class ConsciousnessProcessor(ABC):
    """Abstract base class for consciousness processing"""
    
    @abstractmethod
    async def process_with_consciousness(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with consciousness awareness"""
        pass
    
    @abstractmethod
    async def reflect(self) -> ConsciousnessState:
        """Perform self-reflection and return consciousness state"""
        pass
    
    @abstractmethod
    async def detect_bias(self, decision_data: Dict[str, Any]) -> List[str]:
        """Detect potential biases in decision-making"""
        pass

class NISConsciousnessInterface(ConsciousnessProcessor):
    """
    Practical interface for NIS Protocol v3.0 consciousness capabilities
    Provides easy integration with meta-cognitive processing
    """
    
    def __init__(self, config: ConsciousnessConfig):
        self.config = config
        self.current_state = ConsciousnessState(
            self_awareness_score=0.0,
            cognitive_load=0.0,
            bias_flags=[],
            meta_cognitive_insights=[],
            introspection_depth=config.introspection_depth,
            emotional_state={},
            attention_focus=[]
        )
        self.processing_history = []
        self.reflection_callbacks = []
        
        # Validate configuration
        validation = config.validate()
        if not validation["valid"]:
            raise ValueError(f"Invalid consciousness configuration: {validation['errors']}")
    
    async def process_with_consciousness(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input with full consciousness awareness
        
        Args:
            input_data: Data to process with consciousness
            
        Returns:
            Dictionary containing processed result and consciousness metadata
        """
        
        # Step 1: Pre-processing consciousness assessment
        pre_state = await self.reflect()
        
        # Step 2: Detect any immediate biases in the input
        input_biases = await self.detect_bias(input_data)
        
        # Step 3: Conscious processing with meta-cognitive monitoring
        processing_result = await self._conscious_processing(input_data, pre_state)
        
        # Step 4: Post-processing self-reflection
        post_state = await self.reflect()
        
        # Step 5: Meta-cognitive analysis of the processing
        meta_analysis = await self._meta_cognitive_analysis(
            input_data, processing_result, pre_state, post_state
        )
        
        # Step 6: Update consciousness state
        await self._update_consciousness_state(processing_result, meta_analysis)
        
        # Step 7: Store processing history
        self.processing_history.append({
            "timestamp": asyncio.get_event_loop().time(),
            "input": input_data,
            "result": processing_result,
            "pre_state": pre_state.to_dict(),
            "post_state": post_state.to_dict(),
            "meta_analysis": meta_analysis
        })
        
        return {
            "result": processing_result,
            "consciousness_state": post_state.to_dict(),
            "meta_cognitive_analysis": meta_analysis,
            "bias_flags": input_biases,
            "processing_confidence": meta_analysis.get("confidence", 0.0)
        }
    
    async def reflect(self) -> ConsciousnessState:
        """
        Perform deep self-reflection and return current consciousness state
        
        Returns:
            Current consciousness state with updated metrics
        """
        
        # Analyze recent processing history
        recent_history = self.processing_history[-10:] if self.processing_history else []
        
        # Calculate self-awareness score
        self_awareness = await self._calculate_self_awareness(recent_history)
        
        # Calculate cognitive load
        cognitive_load = await self._calculate_cognitive_load(recent_history)
        
        # Identify current bias flags
        bias_flags = await self._identify_current_biases(recent_history)
        
        # Generate meta-cognitive insights
        insights = await self._generate_meta_cognitive_insights(recent_history)
        
        # Assess emotional state
        emotional_state = await self._assess_emotional_state(recent_history)
        
        # Determine attention focus
        attention_focus = await self._determine_attention_focus(recent_history)
        
        # Update current state
        self.current_state = ConsciousnessState(
            self_awareness_score=self_awareness,
            cognitive_load=cognitive_load,
            bias_flags=bias_flags,
            meta_cognitive_insights=insights,
            introspection_depth=self.config.introspection_depth,
            emotional_state=emotional_state,
            attention_focus=attention_focus
        )
        
        # Trigger reflection callbacks
        for callback in self.reflection_callbacks:
            await callback(self.current_state)
        
        return self.current_state
    
    async def detect_bias(self, decision_data: Dict[str, Any]) -> List[str]:
        """
        Detect potential biases in decision-making
        
        Args:
            decision_data: Data about the decision being made
            
        Returns:
            List of detected bias types
        """
        
        detected_biases = []
        
        # Confirmation bias detection
        if await self._detect_confirmation_bias(decision_data):
            detected_biases.append("confirmation_bias")
        
        # Availability bias detection
        if await self._detect_availability_bias(decision_data):
            detected_biases.append("availability_bias")
        
        # Anchoring bias detection
        if await self._detect_anchoring_bias(decision_data):
            detected_biases.append("anchoring_bias")
        
        # Cultural bias detection
        if await self._detect_cultural_bias(decision_data):
            detected_biases.append("cultural_bias")
        
        # Cognitive overload bias
        if self.current_state.cognitive_load > 0.8:
            detected_biases.append("cognitive_overload")
        
        return detected_biases
    
    def add_reflection_callback(self, callback: Callable[[ConsciousnessState], None]):
        """Add a callback to be called during reflection"""
        self.reflection_callbacks.append(callback)
    
    async def get_consciousness_metrics(self) -> Dict[str, Any]:
        """Get comprehensive consciousness metrics"""
        
        current_state = await self.reflect()
        
        return {
            "consciousness_health": {
                "self_awareness": current_state.self_awareness_score,
                "cognitive_efficiency": 1.0 - current_state.cognitive_load,
                "bias_management": 1.0 - (len(current_state.bias_flags) / 10.0),
                "meta_cognitive_activity": len(current_state.meta_cognitive_insights) / 5.0
            },
            "processing_statistics": {
                "total_processes": len(self.processing_history),
                "average_confidence": sum(
                    p.get("meta_analysis", {}).get("confidence", 0.0) 
                    for p in self.processing_history
                ) / max(1, len(self.processing_history)),
                "bias_detection_rate": sum(
                    1 for p in self.processing_history 
                    if p.get("bias_flags", [])
                ) / max(1, len(self.processing_history))
            },
            "consciousness_trends": {
                "awareness_trend": self._calculate_awareness_trend(),
                "cognitive_load_trend": self._calculate_cognitive_load_trend(),
                "insight_generation_rate": self._calculate_insight_generation_rate()
            }
        }
    
    # Private methods for consciousness processing
    
    async def _conscious_processing(self, input_data: Dict[str, Any], pre_state: ConsciousnessState) -> Dict[str, Any]:
        """Perform conscious processing with meta-cognitive monitoring"""
        
        # Simulate consciousness-aware processing
        # In a real implementation, this would interface with NIS Protocol v3.0
        
        processing_steps = []
        
        # Step 1: Conscious attention allocation
        attention_allocation = await self._allocate_conscious_attention(input_data)
        processing_steps.append({"step": "attention_allocation", "result": attention_allocation})
        
        # Step 2: Meta-cognitive strategy selection
        strategy = await self._select_meta_cognitive_strategy(input_data, pre_state)
        processing_steps.append({"step": "strategy_selection", "result": strategy})
        
        # Step 3: Conscious reasoning with bias monitoring
        reasoning_result = await self._conscious_reasoning(input_data, strategy, pre_state)
        processing_steps.append({"step": "conscious_reasoning", "result": reasoning_result})
        
        # Step 4: Self-monitoring and adjustment
        monitoring_result = await self._self_monitoring(reasoning_result, pre_state)
        processing_steps.append({"step": "self_monitoring", "result": monitoring_result})
        
        return {
            "final_result": reasoning_result,
            "processing_steps": processing_steps,
            "consciousness_applied": True,
            "meta_cognitive_strategy": strategy
        }
    
    async def _meta_cognitive_analysis(self, input_data: Dict[str, Any], 
                                     processing_result: Dict[str, Any],
                                     pre_state: ConsciousnessState,
                                     post_state: ConsciousnessState) -> Dict[str, Any]:
        """Perform meta-cognitive analysis of processing"""
        
        # Analyze processing quality
        quality_score = await self._assess_processing_quality(processing_result)
        
        # Assess confidence
        confidence = await self._assess_confidence(processing_result, pre_state, post_state)
        
        # Identify learning opportunities
        learning_opportunities = await self._identify_learning_opportunities(
            input_data, processing_result, pre_state, post_state
        )
        
        # Generate improvement suggestions
        improvements = await self._generate_improvement_suggestions(
            processing_result, pre_state, post_state
        )
        
        return {
            "quality_score": quality_score,
            "confidence": confidence,
            "learning_opportunities": learning_opportunities,
            "improvement_suggestions": improvements,
            "consciousness_change": {
                "awareness_delta": post_state.self_awareness_score - pre_state.self_awareness_score,
                "cognitive_load_delta": post_state.cognitive_load - pre_state.cognitive_load,
                "new_insights": len(post_state.meta_cognitive_insights) - len(pre_state.meta_cognitive_insights)
            }
        }
    
    async def _update_consciousness_state(self, processing_result: Dict[str, Any], 
                                        meta_analysis: Dict[str, Any]):
        """Update consciousness state based on processing"""
        
        # Update self-awareness based on meta-cognitive insights
        self.current_state.self_awareness_score = min(1.0, 
            self.current_state.self_awareness_score + 0.01 * meta_analysis.get("quality_score", 0.0)
        )
        
        # Update cognitive load
        complexity = len(str(processing_result)) / 1000.0  # Simple complexity measure
        self.current_state.cognitive_load = min(1.0, complexity * 0.1)
        
        # Add new insights
        new_insights = meta_analysis.get("learning_opportunities", [])
        self.current_state.meta_cognitive_insights.extend(new_insights[:3])  # Keep recent insights
        
        # Limit insight history
        if len(self.current_state.meta_cognitive_insights) > 10:
            self.current_state.meta_cognitive_insights = self.current_state.meta_cognitive_insights[-10:]
    
    # Implementation of specific consciousness capabilities
    
    async def _calculate_self_awareness(self, history: List[Dict[str, Any]]) -> float:
        """Calculate current self-awareness score"""
        if not history:
            return 0.5
        
        # Analyze meta-cognitive activity
        meta_activity = sum(
            len(h.get("meta_analysis", {}).get("learning_opportunities", []))
            for h in history
        ) / len(history)
        
        # Analyze bias detection activity
        bias_activity = sum(
            1 for h in history if h.get("bias_flags", [])
        ) / len(history)
        
        # Combine metrics
        return min(1.0, 0.4 + meta_activity * 0.3 + bias_activity * 0.3)
    
    async def _calculate_cognitive_load(self, history: List[Dict[str, Any]]) -> float:
        """Calculate current cognitive load"""
        if not history:
            return 0.0
        
        # Recent processing complexity
        recent_complexity = sum(
            len(str(h.get("result", {}))) / 1000.0
            for h in history[-3:]
        ) / min(3, len(history))
        
        return min(1.0, recent_complexity * 0.1)
    
    async def _identify_current_biases(self, history: List[Dict[str, Any]]) -> List[str]:
        """Identify current active biases"""
        if not history:
            return []
        
        # Collect all bias flags from recent history
        all_biases = []
        for h in history[-5:]:  # Last 5 processes
            all_biases.extend(h.get("bias_flags", []))
        
        # Count occurrences
        bias_counts = {}
        for bias in all_biases:
            bias_counts[bias] = bias_counts.get(bias, 0) + 1
        
        # Return biases that appear frequently
        return [bias for bias, count in bias_counts.items() if count >= 2]
    
    async def _generate_meta_cognitive_insights(self, history: List[Dict[str, Any]]) -> List[str]:
        """Generate meta-cognitive insights from processing history"""
        insights = []
        
        if not history:
            return ["Initial consciousness state established"]
        
        # Analyze patterns
        if len(history) >= 3:
            confidence_trend = [h.get("meta_analysis", {}).get("confidence", 0.0) for h in history[-3:]]
            if all(c1 < c2 for c1, c2 in zip(confidence_trend, confidence_trend[1:])):
                insights.append("Confidence improving through processing")
        
        # Analyze bias detection
        recent_biases = sum(len(h.get("bias_flags", [])) for h in history[-5:])
        if recent_biases > 0:
            insights.append(f"Active bias monitoring detected {recent_biases} potential biases")
        
        # Analyze processing quality
        recent_quality = [h.get("meta_analysis", {}).get("quality_score", 0.0) for h in history[-3:]]
        if recent_quality:
            avg_quality = sum(recent_quality) / len(recent_quality)
            if avg_quality > 0.8:
                insights.append("High quality processing maintained")
        
        return insights[:5]  # Limit to 5 insights
    
    async def _assess_emotional_state(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Assess current emotional state"""
        if not self.config.emotional_awareness:
            return {}
        
        # Simulate emotional state assessment
        # In real implementation, this would analyze processing patterns
        
        emotional_state = {
            "curiosity": 0.7,  # Base curiosity
            "confidence": 0.5,  # Base confidence
            "satisfaction": 0.6,  # Base satisfaction
            "alertness": 0.8  # Base alertness
        }
        
        # Adjust based on recent processing
        if history:
            recent_confidence = sum(
                h.get("meta_analysis", {}).get("confidence", 0.0)
                for h in history[-3:]
            ) / min(3, len(history))
            
            emotional_state["confidence"] = recent_confidence
            emotional_state["satisfaction"] = min(1.0, recent_confidence + 0.2)
        
        return emotional_state
    
    async def _determine_attention_focus(self, history: List[Dict[str, Any]]) -> List[str]:
        """Determine current attention focus areas"""
        if not self.config.attention_tracking:
            return []
        
        focus_areas = []
        
        # Analyze recent processing types
        if history:
            recent_inputs = [h.get("input", {}) for h in history[-5:]]
            
            # Identify common themes
            themes = []
            for inp in recent_inputs:
                if isinstance(inp, dict):
                    themes.extend(inp.keys())
            
            # Count theme frequency
            theme_counts = {}
            for theme in themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            # Select top focus areas
            focus_areas = sorted(theme_counts.keys(), key=lambda x: theme_counts[x], reverse=True)[:3]
        
        if not focus_areas:
            focus_areas = ["consciousness_monitoring", "bias_detection", "meta_cognitive_analysis"]
        
        return focus_areas
    
    # Bias detection implementations
    
    async def _detect_confirmation_bias(self, decision_data: Dict[str, Any]) -> bool:
        """Detect confirmation bias in decision data"""
        # Simplified confirmation bias detection
        if "expected_outcome" in decision_data and "evidence" in decision_data:
            evidence = decision_data["evidence"]
            expected = decision_data["expected_outcome"]
            
            # Check if evidence disproportionately supports expected outcome
            if isinstance(evidence, list) and len(evidence) > 2:
                supporting = sum(1 for e in evidence if "support" in str(e).lower())
                total = len(evidence)
                return supporting / total > 0.8  # More than 80% supporting
        
        return False
    
    async def _detect_availability_bias(self, decision_data: Dict[str, Any]) -> bool:
        """Detect availability bias"""
        # Check if decision is based on easily recalled information
        if "recent_examples" in decision_data:
            recent_weight = decision_data.get("recent_examples_weight", 0.5)
            return recent_weight > 0.7  # Too much weight on recent examples
        
        return False
    
    async def _detect_anchoring_bias(self, decision_data: Dict[str, Any]) -> bool:
        """Detect anchoring bias"""
        # Check if decision is too influenced by initial information
        if "initial_value" in decision_data and "final_value" in decision_data:
            initial = decision_data["initial_value"]
            final = decision_data["final_value"]
            
            if isinstance(initial, (int, float)) and isinstance(final, (int, float)):
                # Check if final value is too close to initial
                difference = abs(final - initial) / max(abs(initial), 1)
                return difference < 0.1  # Less than 10% adjustment
        
        return False
    
    async def _detect_cultural_bias(self, decision_data: Dict[str, Any]) -> bool:
        """Detect cultural bias"""
        # Check for cultural assumptions
        cultural_markers = [
            "cultural_assumption", "stereotype", "generalization",
            "cultural_norm", "traditional", "standard_practice"
        ]
        
        data_str = str(decision_data).lower()
        return any(marker in data_str for marker in cultural_markers)
    
    # Additional helper methods
    
    async def _allocate_conscious_attention(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate conscious attention based on input complexity"""
        complexity = len(str(input_data)) / 100.0
        importance = input_data.get("importance", 0.5)
        
        attention_allocation = {
            "primary_focus": min(1.0, complexity * 0.5 + importance * 0.5),
            "background_monitoring": 0.3,
            "meta_cognitive_overhead": 0.2
        }
        
        return attention_allocation
    
    async def _select_meta_cognitive_strategy(self, input_data: Dict[str, Any], 
                                            pre_state: ConsciousnessState) -> str:
        """Select appropriate meta-cognitive strategy"""
        
        if pre_state.cognitive_load > 0.8:
            return "simplified_processing"
        elif len(pre_state.bias_flags) > 0:
            return "bias_aware_processing"
        elif pre_state.self_awareness_score > 0.8:
            return "high_confidence_processing"
        else:
            return "exploratory_processing"
    
    async def _conscious_reasoning(self, input_data: Dict[str, Any], 
                                 strategy: str, pre_state: ConsciousnessState) -> Dict[str, Any]:
        """Perform conscious reasoning with selected strategy"""
        
        # Simulate conscious reasoning based on strategy
        reasoning_result = {
            "strategy_used": strategy,
            "reasoning_steps": [],
            "conclusion": None,
            "confidence": 0.0
        }
        
        # Add strategy-specific reasoning
        if strategy == "bias_aware_processing":
            reasoning_result["reasoning_steps"].append("Explicitly checked for known biases")
            reasoning_result["confidence"] = 0.8
        elif strategy == "high_confidence_processing":
            reasoning_result["reasoning_steps"].append("Leveraged high self-awareness")
            reasoning_result["confidence"] = 0.9
        elif strategy == "simplified_processing":
            reasoning_result["reasoning_steps"].append("Used simplified approach due to cognitive load")
            reasoning_result["confidence"] = 0.6
        else:
            reasoning_result["reasoning_steps"].append("Explored multiple possibilities")
            reasoning_result["confidence"] = 0.7
        
        # Generate conclusion based on input
        reasoning_result["conclusion"] = f"Processed using {strategy} with confidence {reasoning_result['confidence']}"
        
        return reasoning_result
    
    async def _self_monitoring(self, reasoning_result: Dict[str, Any], 
                             pre_state: ConsciousnessState) -> Dict[str, Any]:
        """Monitor and adjust reasoning process"""
        
        monitoring_result = {
            "quality_check": "passed",
            "adjustments_made": [],
            "confidence_adjustment": 0.0
        }
        
        # Check reasoning quality
        if reasoning_result["confidence"] < 0.5:
            monitoring_result["adjustments_made"].append("Flagged low confidence")
            monitoring_result["confidence_adjustment"] = 0.1
        
        # Check for bias flags
        if pre_state.bias_flags:
            monitoring_result["adjustments_made"].append("Applied bias correction")
            monitoring_result["confidence_adjustment"] -= 0.05
        
        return monitoring_result
    
    # Assessment methods
    
    async def _assess_processing_quality(self, processing_result: Dict[str, Any]) -> float:
        """Assess the quality of processing"""
        
        quality_factors = []
        
        # Check completeness
        if "final_result" in processing_result and processing_result["final_result"]:
            quality_factors.append(0.3)
        
        # Check consciousness application
        if processing_result.get("consciousness_applied", False):
            quality_factors.append(0.3)
        
        # Check meta-cognitive engagement
        if processing_result.get("meta_cognitive_strategy"):
            quality_factors.append(0.2)
        
        # Check processing steps
        if len(processing_result.get("processing_steps", [])) > 2:
            quality_factors.append(0.2)
        
        return sum(quality_factors)
    
    async def _assess_confidence(self, processing_result: Dict[str, Any],
                               pre_state: ConsciousnessState,
                               post_state: ConsciousnessState) -> float:
        """Assess confidence in processing result"""
        
        base_confidence = processing_result.get("final_result", {}).get("confidence", 0.5)
        
        # Adjust based on consciousness state changes
        awareness_improvement = post_state.self_awareness_score - pre_state.self_awareness_score
        confidence_adjustment = awareness_improvement * 0.2
        
        # Adjust based on bias detection
        if pre_state.bias_flags:
            confidence_adjustment -= 0.1
        
        return min(1.0, max(0.0, base_confidence + confidence_adjustment))
    
    async def _identify_learning_opportunities(self, input_data: Dict[str, Any],
                                             processing_result: Dict[str, Any],
                                             pre_state: ConsciousnessState,
                                             post_state: ConsciousnessState) -> List[str]:
        """Identify opportunities for learning and improvement"""
        
        opportunities = []
        
        # Check for bias learning opportunities
        if pre_state.bias_flags:
            opportunities.append("Learn to mitigate detected biases")
        
        # Check for confidence improvement opportunities
        confidence = processing_result.get("final_result", {}).get("confidence", 0.5)
        if confidence < 0.7:
            opportunities.append("Improve confidence through better reasoning")
        
        # Check for meta-cognitive improvement
        if post_state.self_awareness_score <= pre_state.self_awareness_score:
            opportunities.append("Enhance meta-cognitive awareness")
        
        return opportunities
    
    async def _generate_improvement_suggestions(self, processing_result: Dict[str, Any],
                                              pre_state: ConsciousnessState,
                                              post_state: ConsciousnessState) -> List[str]:
        """Generate specific improvement suggestions"""
        
        suggestions = []
        
        # Cognitive load management
        if post_state.cognitive_load > 0.8:
            suggestions.append("Consider breaking complex tasks into smaller parts")
        
        # Bias management
        if len(post_state.bias_flags) > len(pre_state.bias_flags):
            suggestions.append("Implement stronger bias detection protocols")
        
        # Self-awareness enhancement
        if post_state.self_awareness_score < 0.7:
            suggestions.append("Increase frequency of self-reflection activities")
        
        return suggestions
    
    # Trend calculation methods
    
    def _calculate_awareness_trend(self) -> str:
        """Calculate self-awareness trend"""
        if len(self.processing_history) < 5:
            return "insufficient_data"
        
        recent_awareness = [
            h.get("post_state", {}).get("self_awareness_score", 0.0)
            for h in self.processing_history[-5:]
        ]
        
        if len(recent_awareness) < 2:
            return "stable"
        
        trend = sum(recent_awareness[i] - recent_awareness[i-1] for i in range(1, len(recent_awareness)))
        
        if trend > 0.05:
            return "increasing"
        elif trend < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_cognitive_load_trend(self) -> str:
        """Calculate cognitive load trend"""
        if len(self.processing_history) < 5:
            return "insufficient_data"
        
        recent_load = [
            h.get("post_state", {}).get("cognitive_load", 0.0)
            for h in self.processing_history[-5:]
        ]
        
        if len(recent_load) < 2:
            return "stable"
        
        trend = sum(recent_load[i] - recent_load[i-1] for i in range(1, len(recent_load)))
        
        if trend > 0.05:
            return "increasing"
        elif trend < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_insight_generation_rate(self) -> float:
        """Calculate the rate of insight generation"""
        if len(self.processing_history) < 5:
            return 0.0
        
        recent_insights = [
            len(h.get("post_state", {}).get("meta_cognitive_insights", []))
            for h in self.processing_history[-5:]
        ]
        
        if not recent_insights:
            return 0.0
        
        return sum(recent_insights) / len(recent_insights)

# Example usage and integration helpers

class ConsciousnessMiddleware:
    """Middleware for adding consciousness to existing systems"""
    
    def __init__(self, consciousness_interface: NISConsciousnessInterface):
        self.consciousness = consciousness_interface
    
    async def process_with_consciousness_middleware(self, 
                                                   original_processor: Callable,
                                                   input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add consciousness to any existing processor"""
        
        # Pre-process with consciousness
        consciousness_result = await self.consciousness.process_with_consciousness(input_data)
        
        # Run original processor
        original_result = await original_processor(input_data)
        
        # Combine results
        return {
            "original_result": original_result,
            "consciousness_enhancement": consciousness_result,
            "combined_confidence": (
                consciousness_result.get("processing_confidence", 0.0) + 
                original_result.get("confidence", 0.0)
            ) / 2
        }

# Factory function for easy integration
def create_consciousness_interface(consciousness_level: str = "standard") -> NISConsciousnessInterface:
    """Create a consciousness interface with predefined settings"""
    
    if consciousness_level == "high":
        config = ConsciousnessConfig(
            meta_cognitive_processing=True,
            bias_detection=True,
            self_reflection_interval=60,  # 1 minute
            introspection_depth=0.9,
            emotional_awareness=True,
            attention_tracking=True,
            consciousness_threshold=0.8
        )
    elif consciousness_level == "minimal":
        config = ConsciousnessConfig(
            meta_cognitive_processing=True,
            bias_detection=True,
            self_reflection_interval=600,  # 10 minutes
            introspection_depth=0.6,
            emotional_awareness=False,
            attention_tracking=False,
            consciousness_threshold=0.6
        )
    else:  # standard
        config = ConsciousnessConfig()
    
    return NISConsciousnessInterface(config) 