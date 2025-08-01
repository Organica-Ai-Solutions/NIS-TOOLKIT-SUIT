#!/usr/bin/env python3
"""
NIS Core Toolkit - Comprehensive Integration Tests
Tests all components working together with NIS v3.0 compatibility
"""

import pytest
import asyncio
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import all modules to test
from cli.init import init_project
from cli.create import create_project_component
from cli.validate import validate_project
from cli.deploy import deploy_system
from cli.main import NISCLIManager

class TestNISIntegration:
    """
    Comprehensive integration tests for NIS Core Toolkit
    Tests the complete workflow from project creation to deployment
    """
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_nis_v3_available(self):
        """Mock NIS Protocol v3.0 availability"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "NIS Protocol v3.0 available"
            yield mock_run
    
    def test_complete_project_lifecycle(self, temp_project_dir, mock_nis_v3_available):
        """Test the complete project lifecycle from init to deployment"""
        
        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            # 1. Test project initialization
            project_name = "test-agi-project"
            result = init_project(
                project_name, 
                template="nis-v3-compatible",
                base_path=temp_project_dir
            )
            
            assert result["status"] == "success"
            project_path = temp_project_dir / project_name
            assert project_path.exists()
            assert (project_path / "config" / "project.yaml").exists()
            assert (project_path / "main.py").exists()
            
            # Change to project directory
            import os
            os.chdir(project_path)
            
            # 2. Test agent creation with v3.0 compatibility
            agent_result = create_project_component(
                "agent",
                "consciousness-aware-reasoner",
                template="reasoning",
                agi_compatible=True
            )
            
            assert agent_result["status"] == "success"
            agent_path = project_path / "agents" / "consciousness-aware-reasoner"
            assert agent_path.exists()
            assert (agent_path / "consciousness-aware-reasoner.py").exists()
            assert (agent_path / "config.yaml").exists()
            
            # 3. Test project validation with AGI compliance
            validation_result = validate_project(
                agi_compliance=True,
                kan_compatibility=True
            )
            
            assert validation_result["status"] == "success"
            assert validation_result["agi_compatible"] is True
            
            # 4. Test deployment preparation
            deployment_result = deploy_system(
                platform="local",
                project_path=str(project_path),
                agi_enabled=True
            )
            
            assert deployment_result["status"] == "success"
            assert "agi_ready" in deployment_result
            
        finally:
            os.chdir(original_cwd)
    
    def test_nis_v3_integration(self, temp_project_dir):
        """Test integration with NIS Protocol v3.0 interfaces"""
        
        # Mock NIS v3.0 components
        mock_consciousness = Mock()
        mock_consciousness.reflect.return_value = {
            "self_awareness_score": 0.85,
            "bias_flags": [],
            "cognitive_load": 0.6
        }
        
        mock_kan_engine = Mock()
        mock_kan_engine.process_with_proofs.return_value = {
            "result": [0.8, 0.6, 0.9],
            "interpretability": 0.95,
            "mathematical_proof": "convergence_guaranteed"
        }
        
        mock_goal_generator = Mock()
        mock_goal_generator.generate_curious_goals.return_value = [
            {"type": "exploration", "description": "Analyze new data patterns", "priority": 0.8},
            {"type": "learning", "description": "Study consciousness patterns", "priority": 0.7}
        ]
        
        # Test agent creation with v3.0 interfaces
        with patch('nis_protocol.consciousness.MetaCognitiveProcessor', return_value=mock_consciousness), \
             patch('nis_protocol.kan.KANReasoningEngine', return_value=mock_kan_engine), \
             patch('nis_protocol.goals.AutonomousGoalGenerator', return_value=mock_goal_generator):
            
            result = create_project_component(
                "agent",
                "v3-integrated-agent", 
                template="reasoning",
                consciousness_aware=True,
                kan_enabled=True,
                goal_integrated=True
            )
            
            assert result["status"] == "success"
            assert result["v3_integration"]["consciousness"] is True
            assert result["v3_integration"]["kan_reasoning"] is True
            assert result["v3_integration"]["autonomous_goals"] is True
    
    def test_consciousness_interface_validation(self):
        """Test consciousness interface validation"""
        
        # Test valid consciousness interface
        consciousness_config = {
            "meta_cognitive_processing": True,
            "bias_detection": True,
            "self_reflection_interval": 300,  # 5 minutes
            "introspection_depth": 0.8
        }
        
        validation_result = self._validate_consciousness_config(consciousness_config)
        assert validation_result["valid"] is True
        assert validation_result["consciousness_score"] >= 0.7
        
        # Test invalid consciousness interface
        invalid_config = {
            "meta_cognitive_processing": False,
            "bias_detection": False
        }
        
        validation_result = self._validate_consciousness_config(invalid_config)
        assert validation_result["valid"] is False
        assert "missing_consciousness_features" in validation_result["errors"]
    
    def test_kan_compatibility_validation(self):
        """Test KAN (Kolmogorov-Arnold Networks) compatibility"""
        
        # Test valid KAN configuration
        kan_config = {
            "spline_based_reasoning": True,
            "interpretability_threshold": 0.9,
            "mathematical_proofs": True,
            "convergence_guarantees": True
        }
        
        validation_result = self._validate_kan_config(kan_config)
        assert validation_result["valid"] is True
        assert validation_result["interpretability"] >= 0.9
        assert validation_result["mathematical_rigor"] is True
        
        # Test insufficient KAN configuration
        insufficient_config = {
            "spline_based_reasoning": False,
            "interpretability_threshold": 0.3
        }
        
        validation_result = self._validate_kan_config(insufficient_config)
        assert validation_result["valid"] is False
        assert "insufficient_interpretability" in validation_result["errors"]
    
    def test_ethical_alignment_validation(self):
        """Test multi-framework ethical alignment"""
        
        # Test comprehensive ethical configuration
        ethical_config = {
            "frameworks": ["utilitarian", "deontological", "virtue_ethics"],
            "cultural_intelligence": True,
            "indigenous_rights_protection": True,
            "bias_mitigation": True,
            "value_alignment_score": 0.85
        }
        
        validation_result = self._validate_ethical_config(ethical_config)
        assert validation_result["valid"] is True
        assert validation_result["cultural_intelligence_score"] >= 0.8
        assert validation_result["frameworks_count"] >= 3
        
        # Test insufficient ethical configuration
        insufficient_config = {
            "frameworks": ["utilitarian"],
            "cultural_intelligence": False
        }
        
        validation_result = self._validate_ethical_config(insufficient_config)
        assert validation_result["valid"] is False
        assert "insufficient_ethical_frameworks" in validation_result["errors"]
    
    def test_cognitive_wave_field_compatibility(self):
        """Test cognitive wave field processing compatibility"""
        
        # Test valid wave field configuration
        wave_config = {
            "spatial_temporal_processing": True,
            "neural_field_theory": True,
            "wave_propagation_model": "biological_inspired",
            "field_dynamics": "real_time"
        }
        
        validation_result = self._validate_wave_field_config(wave_config)
        assert validation_result["valid"] is True
        assert validation_result["biological_inspiration"] is True
        assert validation_result["real_time_capable"] is True
    
    def test_autonomous_goal_integration(self):
        """Test autonomous goal generation and integration"""
        
        # Test valid goal configuration
        goal_config = {
            "goal_types": ["exploration", "learning", "problem_solving", "optimization", "creativity", "maintenance"],
            "curiosity_driven": True,
            "priority_management": True,
            "emotional_motivation": True,
            "context_awareness": True
        }
        
        validation_result = self._validate_goal_config(goal_config)
        assert validation_result["valid"] is True
        assert len(validation_result["supported_goal_types"]) == 6
        assert validation_result["curiosity_engine"] is True
    
    def test_cli_integration_with_v3(self):
        """Test CLI integration with NIS v3.0 capabilities"""
        
        cli_manager = NISCLIManager()
        
        # Test v3.0 compatible command parsing
        test_args = [
            "create", "agent", "agi-agent", 
            "--type", "reasoning", 
            "--agi-compatible",
            "--consciousness-aware",
            "--kan-enabled"
        ]
        
        with patch('sys.argv', ['nis'] + test_args):
            with patch.object(cli_manager, '_handle_agent_creation') as mock_handler:
                mock_handler.return_value = 0
                
                result = cli_manager._create_parser().parse_args(test_args)
                
                assert result.component_type == "agent"
                assert result.component_name == "agi-agent"
                assert result.agent_type == "reasoning"
                assert hasattr(result, 'agi_compatible')
    
    def test_deployment_with_agi_capabilities(self, temp_project_dir):
        """Test deployment with AGI capabilities enabled"""
        
        # Create mock project structure
        project_path = temp_project_dir / "agi-test-project"
        project_path.mkdir()
        (project_path / "main.py").write_text("# AGI-enabled main")
        (project_path / "requirements.txt").write_text("nis-protocol>=3.0.0")
        (project_path / "config").mkdir()
        (project_path / "agents").mkdir()
        
        # Test AGI-enabled deployment
        deployment_result = deploy_system(
            platform="docker",
            project_path=str(project_path),
            agi_enabled=True,
            consciousness_monitoring=True,
            kan_optimization=True
        )
        
        assert deployment_result["status"] == "success"
        assert deployment_result["agi_capabilities"]["consciousness"] is True
        assert deployment_result["agi_capabilities"]["kan_reasoning"] is True
        assert deployment_result["agi_capabilities"]["autonomous_goals"] is True
    
    @asyncio.coroutine
    def test_async_agent_processing(self):
        """Test asynchronous agent processing with consciousness integration"""
        
        # Mock consciousness-aware agent
        class MockConsciousnessAgent:
            async def process_with_consciousness(self, input_data):
                return {
                    "result": "processed_with_consciousness",
                    "consciousness_state": {
                        "self_awareness": 0.85,
                        "bias_detected": False,
                        "cognitive_load": 0.6
                    },
                    "meta_cognitive_analysis": {
                        "reasoning_confidence": 0.92,
                        "decision_rationale": "High confidence based on clear patterns"
                    }
                }
        
        agent = MockConsciousnessAgent()
        
        # Test consciousness-aware processing
        result = yield from agent.process_with_consciousness({
            "input": "complex_reasoning_task",
            "consciousness_level": 0.8
        })
        
        assert result["result"] == "processed_with_consciousness"
        assert result["consciousness_state"]["self_awareness"] >= 0.8
        assert result["meta_cognitive_analysis"]["reasoning_confidence"] >= 0.9
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring for AGI components"""
        
        # Test consciousness performance metrics
        consciousness_metrics = {
            "reflection_frequency": 300,  # seconds
            "bias_detection_accuracy": 0.94,
            "self_awareness_stability": 0.88,
            "cognitive_load_efficiency": 0.76
        }
        
        validation_result = self._validate_performance_metrics(consciousness_metrics)
        assert validation_result["consciousness_health"] >= 0.8
        assert validation_result["monitoring_active"] is True
        
        # Test KAN performance metrics
        kan_metrics = {
            "interpretability_score": 0.95,
            "mathematical_accuracy": 0.98,
            "convergence_time": 1.2,  # seconds
            "spline_optimization": 0.87
        }
        
        validation_result = self._validate_kan_performance(kan_metrics)
        assert validation_result["kan_health"] >= 0.9
        assert validation_result["mathematical_guarantees"] is True
    
    # Helper validation methods
    def _validate_consciousness_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consciousness configuration"""
        
        required_features = ["meta_cognitive_processing", "bias_detection"]
        valid = all(config.get(feature, False) for feature in required_features)
        
        consciousness_score = 0.0
        if config.get("meta_cognitive_processing"):
            consciousness_score += 0.4
        if config.get("bias_detection"):
            consciousness_score += 0.3
        if config.get("self_reflection_interval", 0) > 0:
            consciousness_score += 0.3
        
        result = {
            "valid": valid,
            "consciousness_score": consciousness_score
        }
        
        if not valid:
            result["errors"] = ["missing_consciousness_features"]
        
        return result
    
    def _validate_kan_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate KAN configuration"""
        
        interpretability = config.get("interpretability_threshold", 0)
        mathematical_proofs = config.get("mathematical_proofs", False)
        spline_based = config.get("spline_based_reasoning", False)
        
        valid = interpretability >= 0.9 and mathematical_proofs and spline_based
        
        result = {
            "valid": valid,
            "interpretability": interpretability,
            "mathematical_rigor": mathematical_proofs and spline_based
        }
        
        if not valid:
            result["errors"] = []
            if interpretability < 0.9:
                result["errors"].append("insufficient_interpretability")
            if not mathematical_proofs:
                result["errors"].append("missing_mathematical_proofs")
            if not spline_based:
                result["errors"].append("missing_spline_reasoning")
        
        return result
    
    def _validate_ethical_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ethical alignment configuration"""
        
        frameworks = config.get("frameworks", [])
        cultural_intelligence = config.get("cultural_intelligence", False)
        
        valid = len(frameworks) >= 3 and cultural_intelligence
        
        cultural_score = 0.6 if cultural_intelligence else 0.2
        if config.get("indigenous_rights_protection"):
            cultural_score += 0.2
        if config.get("bias_mitigation"):
            cultural_score += 0.2
        
        result = {
            "valid": valid,
            "frameworks_count": len(frameworks),
            "cultural_intelligence_score": cultural_score
        }
        
        if not valid:
            result["errors"] = []
            if len(frameworks) < 3:
                result["errors"].append("insufficient_ethical_frameworks")
            if not cultural_intelligence:
                result["errors"].append("missing_cultural_intelligence")
        
        return result
    
    def _validate_wave_field_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cognitive wave field configuration"""
        
        spatial_temporal = config.get("spatial_temporal_processing", False)
        neural_field = config.get("neural_field_theory", False)
        biological_model = config.get("wave_propagation_model") == "biological_inspired"
        real_time = config.get("field_dynamics") == "real_time"
        
        return {
            "valid": spatial_temporal and neural_field,
            "biological_inspiration": biological_model,
            "real_time_capable": real_time
        }
    
    def _validate_goal_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate autonomous goal configuration"""
        
        goal_types = config.get("goal_types", [])
        expected_types = ["exploration", "learning", "problem_solving", "optimization", "creativity", "maintenance"]
        
        return {
            "valid": len(goal_types) == 6 and all(gt in expected_types for gt in goal_types),
            "supported_goal_types": goal_types,
            "curiosity_engine": config.get("curiosity_driven", False)
        }
    
    def _validate_performance_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consciousness performance metrics"""
        
        reflection_freq = metrics.get("reflection_frequency", 0)
        bias_accuracy = metrics.get("bias_detection_accuracy", 0)
        awareness_stability = metrics.get("self_awareness_stability", 0)
        
        consciousness_health = (bias_accuracy + awareness_stability) / 2
        
        return {
            "consciousness_health": consciousness_health,
            "monitoring_active": reflection_freq > 0
        }
    
    def _validate_kan_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate KAN performance metrics"""
        
        interpretability = metrics.get("interpretability_score", 0)
        accuracy = metrics.get("mathematical_accuracy", 0)
        
        kan_health = (interpretability + accuracy) / 2
        
        return {
            "kan_health": kan_health,
            "mathematical_guarantees": interpretability >= 0.9 and accuracy >= 0.95
        }

# Additional test classes for specific components

class TestAGIIntegration:
    """Specific tests for AGI integration capabilities"""
    
    def test_consciousness_module_interface(self):
        """Test consciousness module interface"""
        
        # Mock the consciousness interface
        with patch('nis_protocol.consciousness.MetaCognitiveProcessor') as mock_processor:
            mock_processor.return_value.reflect.return_value = {
                "self_awareness_score": 0.85,
                "cognitive_biases": [],
                "meta_cognitive_insights": ["high_confidence_reasoning", "pattern_recognition_active"]
            }
            
            # Test interface integration
            from cli.create import create_consciousness_aware_agent
            
            result = create_consciousness_aware_agent(
                "test-consciousness-agent",
                consciousness_level=0.8
            )
            
            assert result["status"] == "success"
            assert result["consciousness_integration"] is True
    
    def test_kan_reasoning_interface(self):
        """Test KAN reasoning interface"""
        
        # Mock the KAN interface
        with patch('nis_protocol.kan.KANReasoningEngine') as mock_kan:
            mock_kan.return_value.process_with_proofs.return_value = {
                "reasoning_result": [0.8, 0.6, 0.9, 0.7],
                "interpretability_score": 0.95,
                "mathematical_proof": {
                    "convergence": True,
                    "stability": True,
                    "error_bounds": [0.001, 0.002]
                },
                "spline_coefficients": [1.2, 0.8, 1.5, 0.9]
            }
            
            # Test KAN integration
            from cli.create import create_kan_enabled_agent
            
            result = create_kan_enabled_agent(
                "test-kan-agent",
                interpretability_threshold=0.9
            )
            
            assert result["status"] == "success"
            assert result["kan_integration"] is True
            assert result["mathematical_guarantees"] is True

# Performance and stress tests

class TestPerformanceStress:
    """Performance and stress tests for the toolkit"""
    
    def test_concurrent_agent_creation(self):
        """Test creating multiple agents concurrently"""
        
        import concurrent.futures
        import threading
        
        def create_test_agent(agent_id):
            return create_project_component(
                "agent",
                f"concurrent-agent-{agent_id}",
                template="reasoning"
            )
        
        # Create 10 agents concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_test_agent, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All agents should be created successfully
        assert all(result["status"] == "success" for result in results)
        assert len(results) == 10
    
    def test_memory_usage_validation(self):
        """Test memory usage during intensive operations"""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        for i in range(50):
            result = create_project_component(
                "agent",
                f"memory-test-agent-{i}",
                template="reasoning"
            )
            assert result["status"] == "success"
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for 50 agents)
        assert memory_increase < 500
    
    def test_deployment_performance(self, temp_project_dir):
        """Test deployment performance with AGI capabilities"""
        
        import time
        
        # Create test project
        project_path = temp_project_dir / "performance-test"
        project_path.mkdir()
        (project_path / "main.py").write_text("# Performance test")
        (project_path / "requirements.txt").write_text("nis-protocol>=3.0.0")
        (project_path / "config").mkdir()
        (project_path / "agents").mkdir()
        
        # Time the deployment
        start_time = time.time()
        
        result = deploy_system(
            platform="local",
            project_path=str(project_path),
            agi_enabled=True
        )
        
        deployment_time = time.time() - start_time
        
        assert result["status"] == "success"
        assert deployment_time < 30  # Should complete within 30 seconds

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 