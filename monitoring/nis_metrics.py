#!/usr/bin/env python3
"""
NIS TOOLKIT SUIT v3.2.1 - Comprehensive Metrics Collection
Prometheus metrics for system monitoring, agent performance, and observability
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import defaultdict, Counter
from contextlib import contextmanager

try:
    from prometheus_client import (
        Counter as PrometheusCounter,
        Histogram, 
        Gauge,
        Info,
        Enum,
        start_http_server,
        CollectorRegistry,
        REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    """Configuration for metrics collection"""
    enabled: bool = True
    port: int = 8000
    endpoint: str = "/metrics"
    update_interval: float = 10.0
    retention_hours: int = 24
    labels: Dict[str, str] = None

class NISMetricsCollector:
    """Comprehensive metrics collector for NIS TOOLKIT SUIT"""
    
    def __init__(self, config: MetricConfig = None):
        self.config = config or MetricConfig()
        self.active = False
        self.metrics = {}
        self.agent_states = defaultdict(dict)
        self.system_info = {}
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Install with: pip install prometheus_client")
            return
        
        # Initialize metrics
        self._setup_metrics()
        
        # Background metrics update thread
        self.update_thread = None
        
    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        
        # System Health Metrics
        self.metrics['system_up'] = Gauge(
            'nis_system_up',
            'System health status (1 = up, 0 = down)',
            ['component']
        )
        
        self.metrics['system_info'] = Info(
            'nis_system_info',
            'System information',
            ['version', 'environment', 'build']
        )
        
        # Agent Metrics
        self.metrics['active_agents'] = Gauge(
            'nis_active_agents_total',
            'Number of active agents',
            ['agent_type', 'status']
        )
        
        self.metrics['agent_requests'] = PrometheusCounter(
            'nis_agent_requests_total',
            'Total agent requests',
            ['agent_type', 'agent_id', 'method', 'status']
        )
        
        self.metrics['agent_request_duration'] = Histogram(
            'nis_agent_request_duration_seconds',
            'Agent request duration in seconds',
            ['agent_type', 'agent_id', 'method'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        )
        
        self.metrics['agent_memory_usage'] = Gauge(
            'nis_agent_memory_usage_bytes',
            'Agent memory usage in bytes',
            ['agent_type', 'agent_id']
        )
        
        self.metrics['agent_queue_length'] = Gauge(
            'nis_agent_queue_length',
            'Agent processing queue length',
            ['agent_type', 'agent_id']
        )
        
        self.metrics['agent_confidence_score'] = Gauge(
            'nis_agent_confidence_score',
            'Agent processing confidence score',
            ['agent_type', 'agent_id']
        )
        
        self.metrics['agent_errors'] = PrometheusCounter(
            'nis_agent_errors_total',
            'Total agent errors',
            ['agent_type', 'agent_id', 'error_type']
        )
        
        # Performance Metrics
        self.metrics['processing_throughput'] = Gauge(
            'nis_processing_throughput_requests_per_second',
            'Processing throughput in requests per second',
            ['agent_type']
        )
        
        self.metrics['model_inference_time'] = Histogram(
            'nis_model_inference_duration_seconds',
            'Model inference time in seconds',
            ['model_name', 'agent_type'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Resource Metrics
        self.metrics['cpu_usage'] = Gauge(
            'nis_cpu_usage_percent',
            'CPU usage percentage',
            ['component']
        )
        
        self.metrics['memory_usage'] = Gauge(
            'nis_memory_usage_bytes',
            'Memory usage in bytes',
            ['component', 'type']
        )
        
        self.metrics['disk_usage'] = Gauge(
            'nis_disk_usage_bytes',
            'Disk usage in bytes',
            ['mount_point', 'type']
        )
        
        # Cache Metrics
        self.metrics['cache_hits'] = PrometheusCounter(
            'nis_cache_hits_total',
            'Total cache hits',
            ['cache_type', 'key_prefix']
        )
        
        self.metrics['cache_misses'] = PrometheusCounter(
            'nis_cache_misses_total',
            'Total cache misses',
            ['cache_type', 'key_prefix']
        )
        
        self.metrics['cache_size'] = Gauge(
            'nis_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type']
        )
        
        # Database Metrics (if applicable)
        self.metrics['db_connections'] = Gauge(
            'nis_database_connections_active',
            'Active database connections',
            ['database', 'pool']
        )
        
        self.metrics['db_query_duration'] = Histogram(
            'nis_database_query_duration_seconds',
            'Database query duration in seconds',
            ['database', 'operation'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        # Business Metrics
        self.metrics['user_sessions'] = Gauge(
            'nis_user_sessions_active',
            'Active user sessions',
            ['session_type']
        )
        
        self.metrics['task_completion_rate'] = Gauge(
            'nis_task_completion_rate',
            'Task completion rate (0-1)',
            ['task_type', 'agent_type']
        )
        
        logger.info("Metrics initialized successfully")
    
    def start_metrics_server(self, port: Optional[int] = None):
        """Start Prometheus metrics server"""
        if not PROMETHEUS_AVAILABLE:
            logger.error("Cannot start metrics server - Prometheus client not available")
            return False
        
        port = port or self.config.port
        
        try:
            start_http_server(port)
            self.active = True
            logger.info(f"Metrics server started on port {port}")
            
            # Start background update thread
            self.update_thread = threading.Thread(target=self._update_metrics_loop, daemon=True)
            self.update_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False
    
    def stop(self):
        """Stop metrics collection"""
        self.active = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    # Agent Metrics Methods
    def record_agent_created(self, agent_type: str, agent_id: str):
        """Record agent creation"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['active_agents'].labels(
            agent_type=agent_type, 
            status='active'
        ).inc()
        
        self.agent_states[agent_id] = {
            'type': agent_type,
            'status': 'active',
            'created_at': time.time()
        }
        
        logger.debug(f"Recorded agent creation: {agent_type}/{agent_id}")
    
    def record_agent_destroyed(self, agent_type: str, agent_id: str):
        """Record agent destruction"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['active_agents'].labels(
            agent_type=agent_type, 
            status='active'
        ).dec()
        
        if agent_id in self.agent_states:
            del self.agent_states[agent_id]
        
        logger.debug(f"Recorded agent destruction: {agent_type}/{agent_id}")
    
    def record_agent_request(self, agent_type: str, agent_id: str, method: str, 
                           status: str, duration: float):
        """Record agent request"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['agent_requests'].labels(
            agent_type=agent_type,
            agent_id=agent_id,
            method=method,
            status=status
        ).inc()
        
        self.metrics['agent_request_duration'].labels(
            agent_type=agent_type,
            agent_id=agent_id,
            method=method
        ).observe(duration)
        
        logger.debug(f"Recorded agent request: {agent_type}/{agent_id} - {method} ({status}) - {duration:.3f}s")
    
    def record_agent_error(self, agent_type: str, agent_id: str, error_type: str):
        """Record agent error"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['agent_errors'].labels(
            agent_type=agent_type,
            agent_id=agent_id,
            error_type=error_type
        ).inc()
        
        logger.debug(f"Recorded agent error: {agent_type}/{agent_id} - {error_type}")
    
    def update_agent_metrics(self, agent_type: str, agent_id: str, 
                           memory_bytes: Optional[int] = None,
                           queue_length: Optional[int] = None,
                           confidence_score: Optional[float] = None):
        """Update agent-specific metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        if memory_bytes is not None:
            self.metrics['agent_memory_usage'].labels(
                agent_type=agent_type,
                agent_id=agent_id
            ).set(memory_bytes)
        
        if queue_length is not None:
            self.metrics['agent_queue_length'].labels(
                agent_type=agent_type,
                agent_id=agent_id
            ).set(queue_length)
        
        if confidence_score is not None:
            self.metrics['agent_confidence_score'].labels(
                agent_type=agent_type,
                agent_id=agent_id
            ).set(confidence_score)
    
    # Performance Metrics
    def record_model_inference(self, model_name: str, agent_type: str, duration: float):
        """Record model inference time"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['model_inference_time'].labels(
            model_name=model_name,
            agent_type=agent_type
        ).observe(duration)
    
    def update_throughput(self, agent_type: str, requests_per_second: float):
        """Update processing throughput"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['processing_throughput'].labels(
            agent_type=agent_type
        ).set(requests_per_second)
    
    # System Metrics
    def update_system_health(self, component: str, status: bool):
        """Update system component health"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['system_up'].labels(
            component=component
        ).set(1 if status else 0)
    
    def update_resource_usage(self, component: str, cpu_percent: Optional[float] = None,
                            memory_bytes: Optional[int] = None, memory_type: str = 'rss'):
        """Update resource usage metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        if cpu_percent is not None:
            self.metrics['cpu_usage'].labels(
                component=component
            ).set(cpu_percent)
        
        if memory_bytes is not None:
            self.metrics['memory_usage'].labels(
                component=component,
                type=memory_type
            ).set(memory_bytes)
    
    # Cache Metrics
    def record_cache_hit(self, cache_type: str, key_prefix: str = "default"):
        """Record cache hit"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['cache_hits'].labels(
            cache_type=cache_type,
            key_prefix=key_prefix
        ).inc()
    
    def record_cache_miss(self, cache_type: str, key_prefix: str = "default"):
        """Record cache miss"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['cache_misses'].labels(
            cache_type=cache_type,
            key_prefix=key_prefix
        ).inc()
    
    def update_cache_size(self, cache_type: str, size_bytes: int):
        """Update cache size"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['cache_size'].labels(
            cache_type=cache_type
        ).set(size_bytes)
    
    # Context Managers
    @contextmanager
    def time_agent_request(self, agent_type: str, agent_id: str, method: str):
        """Context manager to time agent requests"""
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception as e:
            status = "error"
            self.record_agent_error(agent_type, agent_id, type(e).__name__)
            raise
        finally:
            duration = time.time() - start_time
            self.record_agent_request(agent_type, agent_id, method, status, duration)
    
    @contextmanager  
    def time_model_inference(self, model_name: str, agent_type: str):
        """Context manager to time model inference"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_model_inference(model_name, agent_type, duration)
    
    # Background Update Methods
    def _update_metrics_loop(self):
        """Background thread to update metrics periodically"""
        while self.active:
            try:
                self._update_system_metrics()
                time.sleep(self.config.update_interval)
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _update_system_metrics(self):
        """Update system-level metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.update_resource_usage("system", cpu_percent=cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.update_resource_usage("system", memory_bytes=memory.used, memory_type="used")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics['disk_usage'].labels(
                mount_point="/",
                type="used"
            ).set(disk.used)
            
            self.metrics['disk_usage'].labels(
                mount_point="/",
                type="free"
            ).set(disk.free)
            
        except ImportError:
            logger.debug("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    # Utility Methods
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        return {
            'active': self.active,
            'active_agents': len(self.agent_states),
            'metrics_count': len(self.metrics),
            'prometheus_available': PROMETHEUS_AVAILABLE
        }

# Global metrics collector instance
_global_collector = None

def get_metrics_collector(config: MetricConfig = None) -> NISMetricsCollector:
    """Get global metrics collector instance"""
    global _global_collector
    
    if _global_collector is None:
        _global_collector = NISMetricsCollector(config)
    
    return _global_collector

# Convenience functions for easy integration
def start_metrics_server(port: int = 8000):
    """Start metrics server"""
    collector = get_metrics_collector()
    return collector.start_metrics_server(port)

def record_agent_request(agent_type: str, agent_id: str, method: str, status: str, duration: float):
    """Record agent request"""
    collector = get_metrics_collector()
    collector.record_agent_request(agent_type, agent_id, method, status, duration)

def time_agent_request(agent_type: str, agent_id: str, method: str):
    """Time agent request context manager"""
    collector = get_metrics_collector()
    return collector.time_agent_request(agent_type, agent_id, method)


if __name__ == "__main__":
    # Example usage
    config = MetricConfig(port=8000, update_interval=5.0)
    collector = NISMetricsCollector(config)
    
    if collector.start_metrics_server():
        logger.info("Metrics server started successfully")
        logger.info(f"Metrics available at http://localhost:8000/metrics")
        
        # Example metrics
        collector.record_agent_created("consciousness", "agent_001")
        collector.record_agent_request("consciousness", "agent_001", "process", "success", 0.15)
        collector.update_agent_metrics("consciousness", "agent_001", memory_bytes=1024*1024, confidence_score=0.95)
        
        try:
            # Keep running
            import signal
            import sys
            
            def signal_handler(sig, frame):
                logger.info("Shutting down metrics server")
                collector.stop()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            logger.info("Press Ctrl+C to stop")
            signal.pause()
            
        except KeyboardInterrupt:
            collector.stop()
    else:
        logger.error("Failed to start metrics server")
