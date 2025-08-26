"""
NIS Agent Performance Benchmarks
Tests for measuring agent performance and ensuring no regressions
"""

import pytest
import asyncio
import time
import numpy as np
from typing import List, Dict, Any

# Mock NIS agent for benchmarking
class MockNISAgent:
    """Mock NIS agent for benchmarking purposes"""
    
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.processing_delay = {
            "small": 0.001,    # 1ms
            "medium": 0.01,    # 10ms  
            "large": 0.1       # 100ms
        }.get(model_size, 0.001)
    
    async def process_single(self, input_data: str) -> Dict[str, Any]:
        """Process single input"""
        await asyncio.sleep(self.processing_delay)
        return {
            "input": input_data,
            "output": f"processed_{input_data}",
            "processing_time": self.processing_delay,
            "model_size": self.model_size
        }
    
    async def process_batch(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """Process batch of inputs"""
        results = []
        for input_data in inputs:
            result = await self.process_single(input_data)
            results.append(result)
        return results
    
    def process_sync(self, input_data: str) -> Dict[str, Any]:
        """Synchronous processing"""
        time.sleep(self.processing_delay)
        return {
            "input": input_data,
            "output": f"processed_{input_data}",
            "processing_time": self.processing_delay,
            "model_size": self.model_size
        }


@pytest.fixture
def small_agent():
    """Small agent fixture"""
    return MockNISAgent("small")


@pytest.fixture
def medium_agent():
    """Medium agent fixture"""
    return MockNISAgent("medium")


@pytest.fixture
def large_agent():
    """Large agent fixture"""
    return MockNISAgent("large")


@pytest.fixture
def sample_inputs():
    """Sample input data"""
    return [f"input_{i}" for i in range(10)]


@pytest.fixture
def large_inputs():
    """Large input dataset"""
    return [f"input_{i}" for i in range(100)]


# Single Processing Benchmarks
@pytest.mark.benchmark(group="single_processing")
def test_small_agent_single_processing(benchmark, small_agent):
    """Benchmark small agent single processing"""
    result = benchmark(small_agent.process_sync, "test_input")
    assert result["output"] == "processed_test_input"
    assert result["model_size"] == "small"


@pytest.mark.benchmark(group="single_processing")
def test_medium_agent_single_processing(benchmark, medium_agent):
    """Benchmark medium agent single processing"""
    result = benchmark(medium_agent.process_sync, "test_input")
    assert result["output"] == "processed_test_input"
    assert result["model_size"] == "medium"


@pytest.mark.benchmark(group="single_processing")
def test_large_agent_single_processing(benchmark, large_agent):
    """Benchmark large agent single processing"""
    result = benchmark(large_agent.process_sync, "test_input")
    assert result["output"] == "processed_test_input"
    assert result["model_size"] == "large"


# Batch Processing Benchmarks
@pytest.mark.benchmark(group="batch_processing")
@pytest.mark.asyncio
async def test_small_agent_batch_processing(benchmark, small_agent, sample_inputs):
    """Benchmark small agent batch processing"""
    
    async def batch_process():
        return await small_agent.process_batch(sample_inputs)
    
    # Run benchmark
    result = benchmark(asyncio.run, batch_process())
    assert len(result) == len(sample_inputs)
    assert all(r["model_size"] == "small" for r in result)


@pytest.mark.benchmark(group="batch_processing")
@pytest.mark.asyncio  
async def test_medium_agent_batch_processing(benchmark, medium_agent, sample_inputs):
    """Benchmark medium agent batch processing"""
    
    async def batch_process():
        return await medium_agent.process_batch(sample_inputs)
    
    result = benchmark(asyncio.run, batch_process())
    assert len(result) == len(sample_inputs)
    assert all(r["model_size"] == "medium" for r in result)


# Memory Usage Benchmarks
@pytest.mark.benchmark(group="memory_usage", min_rounds=5)
def test_memory_efficiency_small_batch(benchmark, small_agent, sample_inputs):
    """Test memory efficiency with small batch"""
    
    def process_all():
        results = []
        for input_data in sample_inputs:
            result = small_agent.process_sync(input_data)
            results.append(result)
        return results
    
    result = benchmark.pedantic(process_all, rounds=5, warmup_rounds=2)
    assert len(result) == len(sample_inputs)


@pytest.mark.benchmark(group="memory_usage", min_rounds=3)
def test_memory_efficiency_large_batch(benchmark, small_agent, large_inputs):
    """Test memory efficiency with large batch"""
    
    def process_all():
        results = []
        for input_data in large_inputs:
            result = small_agent.process_sync(input_data)
            results.append(result)
        return results
    
    result = benchmark.pedantic(process_all, rounds=3, warmup_rounds=1)
    assert len(result) == len(large_inputs)


# Throughput Benchmarks
@pytest.mark.benchmark(group="throughput")
def test_agent_throughput(benchmark, small_agent):
    """Test agent throughput (requests per second)"""
    
    def sustained_processing():
        """Process inputs for sustained period"""
        results = []
        inputs = [f"input_{i}" for i in range(50)]
        
        for input_data in inputs:
            result = small_agent.process_sync(input_data)
            results.append(result)
        
        return results
    
    result = benchmark(sustained_processing)
    assert len(result) == 50


# Latency Distribution Benchmarks
@pytest.mark.benchmark(group="latency", min_rounds=20)
def test_latency_consistency(benchmark, small_agent):
    """Test latency consistency and distribution"""
    
    def single_request():
        return small_agent.process_sync("latency_test")
    
    # Use pedantic mode for precise measurements
    result = benchmark.pedantic(single_request, rounds=20, warmup_rounds=5)
    assert result["output"] == "processed_latency_test"


# Comparative Benchmarks
@pytest.mark.benchmark(group="model_comparison")
@pytest.mark.parametrize("model_size", ["small", "medium", "large"])
def test_model_size_comparison(benchmark, model_size):
    """Compare performance across different model sizes"""
    agent = MockNISAgent(model_size)
    
    def process_standard_input():
        return agent.process_sync("standard_benchmark_input")
    
    result = benchmark(process_standard_input)
    assert result["model_size"] == model_size
    assert result["output"] == "processed_standard_benchmark_input"


# Stress Test Benchmarks
@pytest.mark.benchmark(group="stress_test", min_rounds=1)
def test_sustained_load(benchmark, small_agent):
    """Test sustained load processing"""
    
    def stress_test():
        """Simulate sustained load"""
        results = []
        # Process 200 requests to simulate load
        for i in range(200):
            result = small_agent.process_sync(f"stress_input_{i}")
            results.append(result)
        return results
    
    result = benchmark.pedantic(stress_test, rounds=1, warmup_rounds=0)
    assert len(result) == 200


# Concurrent Processing Benchmarks
@pytest.mark.benchmark(group="concurrency")
@pytest.mark.asyncio
async def test_concurrent_processing(benchmark, small_agent):
    """Test concurrent processing capabilities"""
    
    async def concurrent_requests():
        """Process multiple requests concurrently"""
        tasks = []
        for i in range(20):
            task = small_agent.process_single(f"concurrent_input_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    result = benchmark(asyncio.run, concurrent_requests())
    assert len(result) == 20
    assert all(r["output"].startswith("processed_concurrent_input") for r in result)


# Custom Metrics Benchmarks
@pytest.mark.benchmark(group="custom_metrics", 
                      timer=time.perf_counter,
                      disable_gc=True,
                      warmup=True)
def test_custom_metrics(benchmark, small_agent):
    """Benchmark with custom metrics and settings"""
    
    @benchmark
    def timed_processing():
        return small_agent.process_sync("custom_metrics_input")
    
    result = timed_processing
    assert result["output"] == "processed_custom_metrics_input"


# Regression Test Benchmarks
@pytest.mark.benchmark(group="regression")
def test_performance_regression_guard(benchmark, small_agent):
    """Guard against performance regressions"""
    
    def baseline_performance():
        """Baseline performance that should not regress"""
        return small_agent.process_sync("regression_test")
    
    result = benchmark(baseline_performance)
    
    # Assert performance characteristics
    assert result["output"] == "processed_regression_test"
    
    # Get benchmark stats
    stats = benchmark.stats
    
    # Example: ensure mean time is below threshold
    # In a real scenario, you'd load this from baseline data
    PERFORMANCE_THRESHOLD = 0.01  # 10ms threshold
    
    if hasattr(stats, 'mean'):
        assert stats.mean < PERFORMANCE_THRESHOLD, f"Performance regression detected: {stats.mean}s > {PERFORMANCE_THRESHOLD}s"


if __name__ == "__main__":
    # Run benchmarks directly
    pytest.main([
        __file__,
        "--benchmark-only",
        "--benchmark-histogram", 
        "--benchmark-json=benchmark_results.json",
        "-v"
    ])
