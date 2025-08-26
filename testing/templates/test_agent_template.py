"""
NIS Agent Test Template
Template for creating comprehensive tests for NIS agents
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import your actual agent class here
# from src.agents.your_agent import YourAgent

class TestAgentTemplate:
    """
    Template test class for NIS agents
    
    Replace YourAgent with your actual agent class name
    Customize the tests based on your agent's specific functionality
    """
    
    # Fixtures
    @pytest.fixture
    def agent_config(self):
        """Agent configuration fixture"""
        return {
            'agent_id': 'test_agent',
            'model_name': 'test_model',
            'max_tokens': 1000,
            'temperature': 0.7,
            'timeout': 30.0,
            'retry_attempts': 3
        }
    
    @pytest.fixture
    def agent(self, agent_config):
        """Agent instance fixture"""
        # Replace with your actual agent class
        # return YourAgent(agent_config)
        return MockAgent(agent_config)
    
    @pytest.fixture
    def sample_input(self):
        """Sample input data"""
        return {
            'text': 'Hello, world!',
            'context': {'user_id': '12345'},
            'metadata': {'timestamp': '2024-01-01T00:00:00Z'}
        }
    
    @pytest.fixture
    def expected_output(self):
        """Expected output structure"""
        return {
            'response': 'Hello! How can I help you?',
            'confidence': 0.95,
            'reasoning': 'Standard greeting response',
            'metadata': {'model_used': 'test_model'}
        }
    
    # Initialization Tests
    def test_agent_initialization(self, agent_config):
        """Test agent initialization"""
        agent = MockAgent(agent_config)
        
        assert agent.agent_id == agent_config['agent_id']
        assert agent.config == agent_config
        assert agent.is_initialized is False
    
    def test_agent_initialization_with_missing_config(self):
        """Test agent initialization with missing required config"""
        incomplete_config = {'agent_id': 'test'}
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            MockAgent(incomplete_config)
    
    def test_agent_initialization_with_invalid_config(self):
        """Test agent initialization with invalid config values"""
        invalid_config = {
            'agent_id': '',  # Empty ID should fail
            'max_tokens': -1,  # Negative value should fail
            'temperature': 2.0  # Out of range should fail
        }
        
        with pytest.raises(ValueError):
            MockAgent(invalid_config)
    
    # Setup and Teardown Tests
    @pytest.mark.asyncio
    async def test_agent_setup(self, agent):
        """Test agent setup process"""
        assert agent.is_initialized is False
        
        await agent.initialize()
        
        assert agent.is_initialized is True
        assert hasattr(agent, 'model')
        assert hasattr(agent, 'tokenizer')
    
    @pytest.mark.asyncio
    async def test_agent_setup_failure_handling(self, agent):
        """Test agent setup failure handling"""
        # Mock setup failure
        with patch.object(agent, '_load_model', side_effect=Exception("Model load failed")):
            with pytest.raises(Exception, match="Model load failed"):
                await agent.initialize()
            
            assert agent.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_agent_cleanup(self, agent):
        """Test agent cleanup process"""
        await agent.initialize()
        assert agent.is_initialized is True
        
        await agent.cleanup()
        assert agent.is_initialized is False
    
    # Core Processing Tests
    @pytest.mark.asyncio
    async def test_basic_processing(self, agent, sample_input):
        """Test basic input processing"""
        await agent.initialize()
        
        result = await agent.process(sample_input)
        
        assert 'response' in result
        assert 'confidence' in result
        assert 'metadata' in result
        assert result['confidence'] > 0.0
    
    @pytest.mark.asyncio
    async def test_processing_without_initialization(self, agent, sample_input):
        """Test that processing fails without initialization"""
        with pytest.raises(RuntimeError, match="Agent not initialized"):
            await agent.process(sample_input)
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, agent):
        """Test handling of empty input"""
        await agent.initialize()
        
        result = await agent.process({})
        assert result is not None
        assert 'error' in result or 'response' in result
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, agent):
        """Test handling of invalid input"""
        await agent.initialize()
        
        invalid_inputs = [
            None,
            "string_instead_of_dict",
            {'invalid_key': 'value'},
            {'text': None}
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                await agent.process(invalid_input)
    
    # Batch Processing Tests
    @pytest.mark.asyncio
    async def test_batch_processing(self, agent):
        """Test batch processing capabilities"""
        await agent.initialize()
        
        inputs = [
            {'text': f'Input {i}', 'context': {}} 
            for i in range(5)
        ]
        
        results = await agent.process_batch(inputs)
        
        assert len(results) == len(inputs)
        assert all('response' in result for result in results)
    
    @pytest.mark.asyncio
    async def test_empty_batch_processing(self, agent):
        """Test empty batch processing"""
        await agent.initialize()
        
        results = await agent.process_batch([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_large_batch_processing(self, agent):
        """Test large batch processing"""
        await agent.initialize()
        
        # Test with larger batch
        inputs = [
            {'text': f'Input {i}', 'context': {}} 
            for i in range(100)
        ]
        
        results = await agent.process_batch(inputs)
        assert len(results) == 100
    
    # Error Handling Tests
    @pytest.mark.asyncio
    async def test_timeout_handling(self, agent, sample_input):
        """Test timeout handling"""
        await agent.initialize()
        
        # Mock a slow operation
        with patch.object(agent, '_generate_response', 
                         side_effect=asyncio.TimeoutError("Operation timed out")):
            
            result = await agent.process(sample_input)
            assert 'error' in result
            assert 'timeout' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_model_error_handling(self, agent, sample_input):
        """Test model error handling"""
        await agent.initialize()
        
        # Mock model error
        with patch.object(agent, '_generate_response',
                         side_effect=Exception("Model error")):
            
            result = await agent.process(sample_input)
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, agent, sample_input):
        """Test retry mechanism for transient failures"""
        await agent.initialize()
        
        # Mock transient failure followed by success
        call_count = 0
        def mock_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 attempts
                raise Exception("Transient error")
            return {'response': 'Success after retries'}
        
        with patch.object(agent, '_generate_response', side_effect=mock_response):
            result = await agent.process(sample_input)
            
            assert 'response' in result
            assert result['response'] == 'Success after retries'
            assert call_count == 3  # Should have retried
    
    # Performance Tests
    @pytest.mark.asyncio
    async def test_response_time(self, agent, sample_input):
        """Test response time performance"""
        await agent.initialize()
        
        import time
        start_time = time.time()
        
        result = await agent.process(sample_input)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Should respond within reasonable time (adjust threshold as needed)
        assert response_time < 5.0, f"Response time too slow: {response_time}s"
        assert 'response' in result
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, agent):
        """Test memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        await agent.initialize()
        
        # Process multiple inputs
        for i in range(50):
            await agent.process({'text': f'Input {i}', 'context': {}})
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        max_increase = 100 * 1024 * 1024  # 100MB
        assert memory_increase < max_increase, f"Memory usage too high: {memory_increase} bytes"
    
    # Integration Tests
    @pytest.mark.asyncio
    async def test_agent_state_persistence(self, agent):
        """Test agent state persistence across requests"""
        await agent.initialize()
        
        # First request
        result1 = await agent.process({'text': 'Hello'})
        
        # Second request - should maintain context
        result2 = await agent.process({'text': 'How are you?'})
        
        # Verify state is maintained
        assert agent.request_count == 2
        assert 'response' in result1
        assert 'response' in result2
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, agent):
        """Test concurrent processing capabilities"""
        await agent.initialize()
        
        # Create concurrent tasks
        tasks = []
        for i in range(10):
            task = agent.process({'text': f'Concurrent input {i}', 'context': {}})
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        assert len(results) == 10
        assert all('response' in result for result in results if not isinstance(result, Exception))
    
    # Configuration Tests
    def test_config_validation(self, agent_config):
        """Test configuration validation"""
        # Test valid config
        agent = MockAgent(agent_config)
        assert agent.config == agent_config
        
        # Test invalid config values
        invalid_configs = [
            {**agent_config, 'max_tokens': 0},  # Zero tokens
            {**agent_config, 'temperature': -1},  # Invalid temperature
            {**agent_config, 'timeout': 0},  # Zero timeout
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                MockAgent(invalid_config)
    
    @pytest.mark.asyncio
    async def test_config_updates(self, agent):
        """Test dynamic configuration updates"""
        await agent.initialize()
        
        original_temp = agent.config.get('temperature', 0.7)
        new_temp = 0.9
        
        # Update configuration
        agent.update_config({'temperature': new_temp})
        
        assert agent.config['temperature'] == new_temp
        assert agent.config['temperature'] != original_temp


# Mock Agent for Testing
class MockAgent:
    """Mock agent for testing purposes"""
    
    def __init__(self, config: Dict[str, Any]):
        self._validate_config(config)
        self.config = config
        self.agent_id = config['agent_id']
        self.is_initialized = False
        self.model = None
        self.tokenizer = None
        self.request_count = 0
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration"""
        required_keys = ['agent_id', 'model_name']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration: {key}")
        
        if not config['agent_id']:
            raise ValueError("Agent ID cannot be empty")
        
        if config.get('max_tokens', 1) <= 0:
            raise ValueError("max_tokens must be positive")
        
        temp = config.get('temperature', 0.7)
        if temp < 0 or temp > 2.0:
            raise ValueError("temperature must be between 0 and 2.0")
    
    async def initialize(self):
        """Initialize the agent"""
        await self._load_model()
        self.is_initialized = True
    
    async def _load_model(self):
        """Mock model loading"""
        await asyncio.sleep(0.1)  # Simulate loading time
        self.model = Mock()
        self.tokenizer = Mock()
    
    async def cleanup(self):
        """Cleanup agent resources"""
        self.is_initialized = False
        self.model = None
        self.tokenizer = None
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data"""
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized")
        
        if not input_data:
            return {'error': 'Empty input'}
        
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")
        
        self.request_count += 1
        
        # Mock processing
        return await self._generate_response(input_data)
    
    async def _generate_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock response"""
        await asyncio.sleep(0.01)  # Simulate processing time
        
        return {
            'response': f"Processed: {input_data.get('text', 'No text')}",
            'confidence': 0.95,
            'reasoning': 'Mock reasoning',
            'metadata': {
                'model_used': self.config['model_name'],
                'request_id': self.request_count
            }
        }
    
    async def process_batch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of inputs"""
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized")
        
        results = []
        for input_data in inputs:
            result = await self.process(input_data)
            results.append(result)
        
        return results
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update agent configuration"""
        self.config.update(new_config)
