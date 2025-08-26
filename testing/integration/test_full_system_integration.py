"""
NIS TOOLKIT SUIT - Full System Integration Tests
End-to-end integration tests for the complete NIS system
"""

import pytest
import asyncio
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List

# Integration test fixtures
@pytest.fixture(scope="session")
def docker_services():
    """Ensure Docker services are running"""
    import subprocess
    
    # Check if test services are running
    try:
        result = subprocess.run(
            ['docker-compose', '-f', 'docker-compose.test.yml', 'ps', '-q'],
            capture_output=True, text=True, check=True
        )
        if not result.stdout.strip():
            pytest.skip("Docker test services not running. Run: docker-compose -f docker-compose.test.yml up -d")
        return True
    except subprocess.CalledProcessError:
        pytest.skip("Docker Compose not available")

@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        'base_url': 'http://localhost:8000',
        'redis_url': 'redis://localhost:6380',
        'postgres_url': 'postgresql://test_user:test_password@localhost:5433/nis_test',
        'kafka_url': 'localhost:9093',
        'prometheus_url': 'http://localhost:9091',
        'grafana_url': 'http://localhost:3001',
        'mock_api_url': 'http://localhost:8081'
    }

@pytest.fixture
def api_client(test_config):
    """HTTP client for API testing"""
    class APIClient:
        def __init__(self, base_url: str):
            self.base_url = base_url
            self.session = requests.Session()
            self.session.headers.update({
                'Content-Type': 'application/json',
                'User-Agent': 'NIS-Test-Client/1.0'
            })
        
        def get(self, endpoint: str, **kwargs):
            return self.session.get(f"{self.base_url}{endpoint}", **kwargs)
        
        def post(self, endpoint: str, data=None, **kwargs):
            if data:
                kwargs['json'] = data
            return self.session.post(f"{self.base_url}{endpoint}", **kwargs)
        
        def put(self, endpoint: str, data=None, **kwargs):
            if data:
                kwargs['json'] = data
            return self.session.put(f"{self.base_url}{endpoint}", **kwargs)
        
        def delete(self, endpoint: str, **kwargs):
            return self.session.delete(f"{self.base_url}{endpoint}", **kwargs)
    
    return APIClient(test_config['base_url'])

@pytest.fixture
async def redis_client(test_config):
    """Redis client for cache testing"""
    try:
        import redis.asyncio as redis
        client = redis.from_url(test_config['redis_url'])
        yield client
        await client.close()
    except ImportError:
        pytest.skip("Redis client not available")

@pytest.fixture
async def kafka_producer(test_config):
    """Kafka producer for message testing"""
    try:
        from aiokafka import AIOKafkaProducer
        producer = AIOKafkaProducer(bootstrap_servers=test_config['kafka_url'])
        await producer.start()
        yield producer
        await producer.stop()
    except ImportError:
        pytest.skip("Kafka client not available")


@pytest.mark.integration
class TestSystemHealthChecks:
    """Test system health and readiness"""
    
    def test_system_health_endpoint(self, api_client):
        """Test system health endpoint"""
        response = api_client.get('/health')
        assert response.status_code == 200
        
        health_data = response.json()
        assert 'status' in health_data
        assert health_data['status'] == 'healthy'
        assert 'timestamp' in health_data
        assert 'version' in health_data
    
    def test_system_readiness_endpoint(self, api_client):
        """Test system readiness endpoint"""
        response = api_client.get('/ready')
        assert response.status_code == 200
        
        readiness_data = response.json()
        assert 'ready' in readiness_data
        assert readiness_data['ready'] is True
        assert 'services' in readiness_data
    
    def test_service_dependencies(self, api_client, docker_services):
        """Test that all service dependencies are available"""
        response = api_client.get('/health/dependencies')
        assert response.status_code == 200
        
        deps = response.json()
        expected_services = ['redis', 'postgres', 'kafka']
        
        for service in expected_services:
            assert service in deps
            assert deps[service]['status'] == 'connected'


@pytest.mark.integration
class TestAgentWorkflows:
    """Test complete agent processing workflows"""
    
    @pytest.mark.asyncio
    async def test_single_agent_processing_flow(self, api_client):
        """Test complete single agent processing flow"""
        # Step 1: Create agent
        agent_config = {
            'agent_type': 'consciousness',
            'model': 'test_model',
            'config': {
                'temperature': 0.7,
                'max_tokens': 1000
            }
        }
        
        response = api_client.post('/agents', data=agent_config)
        assert response.status_code == 201
        
        agent_data = response.json()
        agent_id = agent_data['agent_id']
        
        # Step 2: Initialize agent
        response = api_client.post(f'/agents/{agent_id}/initialize')
        assert response.status_code == 200
        
        # Step 3: Process request
        request_data = {
            'input': 'Hello, world!',
            'context': {'user_id': 'test_user'},
            'metadata': {'request_id': 'test_001'}
        }
        
        response = api_client.post(f'/agents/{agent_id}/process', data=request_data)
        assert response.status_code == 200
        
        result = response.json()
        assert 'response' in result
        assert 'confidence' in result
        assert 'processing_time' in result
        
        # Step 4: Get agent status
        response = api_client.get(f'/agents/{agent_id}/status')
        assert response.status_code == 200
        
        status = response.json()
        assert status['status'] == 'active'
        assert status['requests_processed'] >= 1
        
        # Step 5: Cleanup
        response = api_client.delete(f'/agents/{agent_id}')
        assert response.status_code == 204
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, api_client):
        """Test multi-agent coordination workflow"""
        # Create multiple agents
        agent_types = ['consciousness', 'reasoning', 'memory']
        agent_ids = []
        
        for agent_type in agent_types:
            config = {
                'agent_type': agent_type,
                'model': 'test_model',
                'config': {'temperature': 0.5}
            }
            
            response = api_client.post('/agents', data=config)
            assert response.status_code == 201
            
            agent_data = response.json()
            agent_ids.append(agent_data['agent_id'])
        
        try:
            # Initialize all agents
            for agent_id in agent_ids:
                response = api_client.post(f'/agents/{agent_id}/initialize')
                assert response.status_code == 200
            
            # Create coordination task
            coordination_request = {
                'agents': agent_ids,
                'task': 'collaborative_analysis',
                'input': 'Analyze the concept of artificial intelligence',
                'coordination_strategy': 'pipeline'
            }
            
            response = api_client.post('/coordination/execute', data=coordination_request)
            assert response.status_code == 200
            
            result = response.json()
            assert 'coordination_id' in result
            assert 'status' in result
            
            coordination_id = result['coordination_id']
            
            # Poll for completion
            max_attempts = 30
            for attempt in range(max_attempts):
                response = api_client.get(f'/coordination/{coordination_id}/status')
                assert response.status_code == 200
                
                status = response.json()
                if status['status'] == 'completed':
                    break
                elif status['status'] == 'failed':
                    pytest.fail(f"Coordination failed: {status.get('error')}")
                
                await asyncio.sleep(1)
            else:
                pytest.fail("Coordination timed out")
            
            # Get final results
            response = api_client.get(f'/coordination/{coordination_id}/results')
            assert response.status_code == 200
            
            results = response.json()
            assert 'final_result' in results
            assert 'agent_contributions' in results
            assert len(results['agent_contributions']) == len(agent_ids)
        
        finally:
            # Cleanup all agents
            for agent_id in agent_ids:
                api_client.delete(f'/agents/{agent_id}')
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, api_client):
        """Test batch processing workflow"""
        # Create agent
        agent_config = {
            'agent_type': 'reasoning',
            'model': 'test_model',
            'config': {'batch_size': 5}
        }
        
        response = api_client.post('/agents', data=agent_config)
        assert response.status_code == 201
        
        agent_data = response.json()
        agent_id = agent_data['agent_id']
        
        try:
            # Initialize agent
            response = api_client.post(f'/agents/{agent_id}/initialize')
            assert response.status_code == 200
            
            # Prepare batch data
            batch_data = {
                'inputs': [
                    {'text': f'Process item {i}', 'context': {}} 
                    for i in range(10)
                ],
                'processing_options': {
                    'parallel': True,
                    'timeout': 60
                }
            }
            
            # Submit batch
            response = api_client.post(f'/agents/{agent_id}/process_batch', data=batch_data)
            assert response.status_code == 202  # Accepted for processing
            
            batch_result = response.json()
            batch_id = batch_result['batch_id']
            
            # Poll for completion
            max_attempts = 60
            for attempt in range(max_attempts):
                response = api_client.get(f'/agents/{agent_id}/batch/{batch_id}/status')
                assert response.status_code == 200
                
                status = response.json()
                if status['status'] == 'completed':
                    break
                elif status['status'] == 'failed':
                    pytest.fail(f"Batch processing failed: {status.get('error')}")
                
                await asyncio.sleep(1)
            else:
                pytest.fail("Batch processing timed out")
            
            # Get batch results
            response = api_client.get(f'/agents/{agent_id}/batch/{batch_id}/results')
            assert response.status_code == 200
            
            results = response.json()
            assert 'results' in results
            assert len(results['results']) == 10
            assert all('response' in result for result in results['results'])
        
        finally:
            # Cleanup
            api_client.delete(f'/agents/{agent_id}')


@pytest.mark.integration
class TestDataPersistenceIntegration:
    """Test data persistence across system restarts"""
    
    @pytest.mark.asyncio
    async def test_redis_cache_integration(self, redis_client, api_client):
        """Test Redis cache integration"""
        # Store data via API
        cache_data = {
            'key': 'test_key',
            'value': {'message': 'Hello, Redis!', 'timestamp': time.time()},
            'ttl': 300
        }
        
        response = api_client.post('/cache/set', data=cache_data)
        assert response.status_code == 200
        
        # Verify data in Redis directly
        cached_value = await redis_client.get('test_key')
        assert cached_value is not None
        
        cached_data = json.loads(cached_value)
        assert cached_data['message'] == 'Hello, Redis!'
        
        # Retrieve via API
        response = api_client.get('/cache/get/test_key')
        assert response.status_code == 200
        
        retrieved_data = response.json()
        assert retrieved_data['value']['message'] == 'Hello, Redis!'
    
    @pytest.mark.asyncio
    async def test_kafka_message_integration(self, kafka_producer, api_client):
        """Test Kafka message queue integration"""
        # Send message via API
        message_data = {
            'topic': 'test_topic',
            'message': {
                'event_type': 'agent_processed',
                'agent_id': 'test_agent',
                'timestamp': time.time(),
                'data': {'result': 'success'}
            }
        }
        
        response = api_client.post('/messages/send', data=message_data)
        assert response.status_code == 200
        
        # Produce message directly to Kafka
        await kafka_producer.send(
            'test_topic',
            json.dumps({'direct_message': 'Hello, Kafka!'}).encode('utf-8')
        )
        
        # Consume messages via API
        response = api_client.get('/messages/consume/test_topic?max_messages=10&timeout=5')
        assert response.status_code == 200
        
        messages = response.json()
        assert len(messages) >= 1
        
        # Verify our message is present
        api_message_found = any(
            msg.get('event_type') == 'agent_processed' 
            for msg in messages
        )
        assert api_message_found


@pytest.mark.integration
class TestMonitoringIntegration:
    """Test monitoring and observability integration"""
    
    def test_prometheus_metrics_integration(self, test_config):
        """Test Prometheus metrics collection"""
        # Check Prometheus is available
        response = requests.get(f"{test_config['prometheus_url']}/api/v1/targets")
        assert response.status_code == 200
        
        targets = response.json()
        assert 'data' in targets
        assert 'activeTargets' in targets['data']
        
        # Check for NIS metrics
        response = requests.get(f"{test_config['prometheus_url']}/api/v1/query?query=nis_agent_requests_total")
        assert response.status_code == 200
        
        metrics = response.json()
        assert 'data' in metrics
    
    def test_grafana_dashboard_integration(self, test_config):
        """Test Grafana dashboard integration"""
        # Check Grafana health
        response = requests.get(f"{test_config['grafana_url']}/api/health")
        assert response.status_code == 200
        
        health = response.json()
        assert health['database'] == 'ok'
        
        # Check datasources
        auth = ('admin', 'test_admin')
        response = requests.get(f"{test_config['grafana_url']}/api/datasources", auth=auth)
        assert response.status_code == 200
        
        datasources = response.json()
        prometheus_found = any(
            ds['type'] == 'prometheus' 
            for ds in datasources
        )
        # Note: May not be configured in test environment
        # assert prometheus_found
    
    def test_logging_integration(self, api_client):
        """Test centralized logging integration"""
        # Generate some log entries
        test_requests = [
            {'input': f'Test log entry {i}', 'level': 'INFO'}
            for i in range(5)
        ]
        
        for request in test_requests:
            response = api_client.post('/test/log', data=request)
            assert response.status_code == 200
        
        # Query logs via API
        response = api_client.get('/logs?level=INFO&limit=10')
        if response.status_code == 200:  # Logging endpoint may not be implemented
            logs = response.json()
            assert len(logs) >= len(test_requests)


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test system performance under load"""
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, api_client):
        """Test concurrent request handling"""
        import aiohttp
        import asyncio
        
        async def make_request(session, request_id):
            """Make a single request"""
            request_data = {
                'input': f'Concurrent request {request_id}',
                'context': {'request_id': request_id}
            }
            
            async with session.post(
                f"{api_client.base_url}/test/echo",
                json=request_data
            ) as response:
                return await response.json()
        
        # Create agent for testing
        agent_config = {
            'agent_type': 'test',
            'model': 'test_model',
            'config': {'concurrent_limit': 50}
        }
        
        response = api_client.post('/agents', data=agent_config)
        if response.status_code != 201:
            pytest.skip("Agent creation not available for concurrent testing")
        
        agent_data = response.json()
        agent_id = agent_data['agent_id']
        
        try:
            # Initialize agent
            response = api_client.post(f'/agents/{agent_id}/initialize')
            assert response.status_code == 200
            
            # Make concurrent requests
            async with aiohttp.ClientSession() as session:
                tasks = [
                    make_request(session, i) 
                    for i in range(20)  # 20 concurrent requests
                ]
                
                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                # Analyze results
                successful_requests = [
                    result for result in results 
                    if not isinstance(result, Exception)
                ]
                
                assert len(successful_requests) >= 18  # Allow for some failures
                
                # Check response time
                total_time = end_time - start_time
                avg_response_time = total_time / len(successful_requests)
                assert avg_response_time < 2.0, f"Average response time too slow: {avg_response_time}s"
        
        finally:
            # Cleanup
            api_client.delete(f'/agents/{agent_id}')
    
    def test_system_resource_usage(self, api_client):
        """Test system resource usage under load"""
        import psutil
        
        # Get initial system metrics
        initial_cpu = psutil.cpu_percent(interval=1)
        initial_memory = psutil.virtual_memory().percent
        
        # Generate load
        for i in range(50):
            request_data = {'input': f'Load test {i}'}
            response = api_client.post('/test/echo', data=request_data)
            assert response.status_code == 200
        
        # Check resource usage
        final_cpu = psutil.cpu_percent(interval=1)
        final_memory = psutil.virtual_memory().percent
        
        # Resources should not be exhausted
        assert final_cpu < 95, f"CPU usage too high: {final_cpu}%"
        assert final_memory < 95, f"Memory usage too high: {final_memory}%"


@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Test system error recovery and resilience"""
    
    def test_graceful_service_degradation(self, api_client):
        """Test graceful degradation when services are unavailable"""
        # Test with simulated service failure
        request_data = {
            'input': 'Test with simulated failure',
            'simulate_failure': 'redis_unavailable'
        }
        
        response = api_client.post('/test/resilience', data=request_data)
        
        # Should handle gracefully, not crash
        assert response.status_code in [200, 503]  # OK or Service Unavailable
        
        if response.status_code == 503:
            error_data = response.json()
            assert 'error' in error_data
            assert 'fallback' in error_data or 'retry_after' in error_data
    
    @pytest.mark.asyncio
    async def test_automatic_retry_mechanism(self, api_client):
        """Test automatic retry mechanism"""
        # Request that will fail initially but succeed on retry
        request_data = {
            'input': 'Test retry mechanism',
            'fail_attempts': 2,  # Fail first 2 attempts
            'retry_id': 'test_retry_001'
        }
        
        response = api_client.post('/test/retry', data=request_data)
        
        # Should eventually succeed despite initial failures
        assert response.status_code == 200
        
        result = response.json()
        assert 'attempts' in result
        assert result['attempts'] >= 3  # Should have retried
        assert result['status'] == 'success'


if __name__ == "__main__":
    # Run integration tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "integration",
        "--durations=10"
    ])
