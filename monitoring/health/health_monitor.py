#!/usr/bin/env python3
"""
NIS TOOLKIT SUIT v4.0.0 - Comprehensive Health Monitoring
Automated health checks, service monitoring, and system diagnostics
"""

import asyncio
import aiohttp
import time
import logging
import threading
import subprocess
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import json

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Individual health check definition"""
    name: str
    description: str
    check_function: Callable
    interval: float = 30.0  # seconds
    timeout: float = 10.0   # seconds
    critical: bool = False
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthResult:
    """Health check result"""
    name: str
    status: HealthStatus
    message: str
    duration: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class SystemHealth:
    """Overall system health"""
    status: HealthStatus
    timestamp: datetime
    checks: Dict[str, HealthResult]
    summary: Dict[str, int] = field(default_factory=dict)
    uptime: float = 0.0

class NISHealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthResult] = {}
        self.running = False
        self.start_time = time.time()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Background monitoring task
        self.monitor_task = None
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks"""
        
        # System checks
        self.register_check(HealthCheck(
            name="system_cpu",
            description="System CPU usage",
            check_function=self._check_cpu_usage,
            interval=15.0,
            critical=True,
            tags=["system", "cpu"]
        ))
        
        self.register_check(HealthCheck(
            name="system_memory", 
            description="System memory usage",
            check_function=self._check_memory_usage,
            interval=15.0,
            critical=True,
            tags=["system", "memory"]
        ))
        
        self.register_check(HealthCheck(
            name="system_disk",
            description="System disk usage",
            check_function=self._check_disk_usage,
            interval=60.0,
            critical=True,
            tags=["system", "disk"]
        ))
        
        # Network checks
        self.register_check(HealthCheck(
            name="network_connectivity",
            description="Network connectivity",
            check_function=self._check_network_connectivity,
            interval=30.0,
            critical=True,
            tags=["network", "connectivity"]
        ))
        
        # Service checks (if available)
        if REDIS_AVAILABLE:
            self.register_check(HealthCheck(
                name="redis_connection",
                description="Redis connection",
                check_function=self._check_redis_connection,
                interval=20.0,
                critical=False,
                tags=["redis", "cache"]
            ))
        
        if POSTGRES_AVAILABLE:
            self.register_check(HealthCheck(
                name="postgres_connection",
                description="PostgreSQL connection", 
                check_function=self._check_postgres_connection,
                interval=20.0,
                critical=False,
                tags=["postgres", "database"]
            ))
        
        if KAFKA_AVAILABLE:
            self.register_check(HealthCheck(
                name="kafka_connection",
                description="Kafka connection",
                check_function=self._check_kafka_connection,
                interval=30.0,
                critical=False,
                tags=["kafka", "messaging"]
            ))
        
        # Application checks
        self.register_check(HealthCheck(
            name="agent_registry",
            description="Agent registry health",
            check_function=self._check_agent_registry,
            interval=10.0,
            critical=True,
            tags=["agents", "registry"]
        ))
        
        self.register_check(HealthCheck(
            name="model_availability",
            description="ML model availability",
            check_function=self._check_model_availability,
            interval=60.0,
            critical=False,
            tags=["models", "ai"]
        ))
    
    def register_check(self, health_check: HealthCheck):
        """Register a health check"""
        self.checks[health_check.name] = health_check
        self.logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_check(self, name: str):
        """Unregister a health check"""
        if name in self.checks:
            del self.checks[name]
            if name in self.results:
                del self.results[name]
            self.logger.info(f"Unregistered health check: {name}")
    
    async def start_monitoring(self):
        """Start health monitoring"""
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Create monitoring task
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        check_schedules = {name: 0 for name in self.checks.keys()}
        
        while self.running:
            current_time = time.time()
            
            # Run due checks
            tasks = []
            for name, check in self.checks.items():
                if not check.enabled:
                    continue
                
                if current_time >= check_schedules[name]:
                    tasks.append(self._run_check(check))
                    check_schedules[name] = current_time + check.interval
            
            # Execute checks concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Sleep for a short interval
            await asyncio.sleep(1)
    
    async def _run_check(self, check: HealthCheck) -> HealthResult:
        """Run a single health check"""
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check.check_function(),
                timeout=check.timeout
            )
            
            duration = time.time() - start_time
            
            if isinstance(result, HealthResult):
                health_result = result
                health_result.duration = duration
            elif isinstance(result, bool):
                health_result = HealthResult(
                    name=check.name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="OK" if result else "Check failed",
                    duration=duration,
                    timestamp=datetime.utcnow()
                )
            else:
                health_result = HealthResult(
                    name=check.name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    duration=duration,
                    timestamp=datetime.utcnow()
                )
        
        except asyncio.TimeoutError:
            health_result = HealthResult(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {check.timeout}s",
                duration=time.time() - start_time,
                timestamp=datetime.utcnow(),
                error="timeout"
            )
        
        except Exception as e:
            health_result = HealthResult(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration=time.time() - start_time,
                timestamp=datetime.utcnow(),
                error=type(e).__name__
            )
        
        # Store result
        self.results[check.name] = health_result
        
        # Log result
        level = logging.ERROR if health_result.status == HealthStatus.UNHEALTHY else logging.DEBUG
        self.logger.log(level, f"Health check '{check.name}': {health_result.status.value} - {health_result.message}")
        
        return health_result
    
    async def run_check_once(self, name: str) -> Optional[HealthResult]:
        """Run a specific health check once"""
        if name not in self.checks:
            return None
        
        return await self._run_check(self.checks[name])
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health"""
        if not self.results:
            return SystemHealth(
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.utcnow(),
                checks={},
                uptime=time.time() - self.start_time
            )
        
        # Calculate summary
        summary = {
            "healthy": 0,
            "degraded": 0, 
            "unhealthy": 0,
            "unknown": 0,
            "total": len(self.results)
        }
        
        for result in self.results.values():
            summary[result.status.value] += 1
        
        # Determine overall status
        if summary["unhealthy"] > 0:
            # Check if any critical checks are unhealthy
            critical_unhealthy = any(
                result.status == HealthStatus.UNHEALTHY and self.checks[result.name].critical
                for result in self.results.values()
            )
            overall_status = HealthStatus.UNHEALTHY if critical_unhealthy else HealthStatus.DEGRADED
        elif summary["degraded"] > 0:
            overall_status = HealthStatus.DEGRADED
        elif summary["healthy"] > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow(),
            checks=self.results.copy(),
            summary=summary,
            uptime=time.time() - self.start_time
        )
    
    def get_health_json(self) -> str:
        """Get system health as JSON"""
        health = self.get_system_health()
        
        # Convert to serializable format
        health_dict = asdict(health)
        health_dict['status'] = health.status.value
        
        for check_name, result in health_dict['checks'].items():
            result['status'] = result['status']
            result['timestamp'] = result['timestamp'].isoformat()
        
        health_dict['timestamp'] = health.timestamp.isoformat()
        
        return json.dumps(health_dict, indent=2)
    
    # Built-in health check implementations
    async def _check_cpu_usage(self) -> HealthResult:
        """Check CPU usage"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High CPU usage: {cpu_percent}%"
            elif cpu_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Elevated CPU usage: {cpu_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage: {cpu_percent}%"
            
            return HealthResult(
                name="system_cpu",
                status=status,
                message=message,
                duration=0,
                timestamp=datetime.utcnow(),
                details={"cpu_percent": cpu_percent}
            )
        
        except ImportError:
            return HealthResult(
                name="system_cpu",
                status=HealthStatus.UNKNOWN,
                message="psutil not available",
                duration=0,
                timestamp=datetime.utcnow()
            )
    
    async def _check_memory_usage(self) -> HealthResult:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High memory usage: {memory.percent}%"
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Elevated memory usage: {memory.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage: {memory.percent}%"
            
            return HealthResult(
                name="system_memory",
                status=status,
                message=message,
                duration=0,
                timestamp=datetime.utcnow(),
                details={
                    "memory_percent": memory.percent,
                    "memory_available": memory.available,
                    "memory_total": memory.total
                }
            )
        
        except ImportError:
            return HealthResult(
                name="system_memory",
                status=HealthStatus.UNKNOWN,
                message="psutil not available",
                duration=0,
                timestamp=datetime.utcnow()
            )
    
    async def _check_disk_usage(self) -> HealthResult:
        """Check disk usage"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            percent = (disk.used / disk.total) * 100
            
            if percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {percent:.1f}%"
            elif percent > 85:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage: {percent:.1f}%"
            
            return HealthResult(
                name="system_disk",
                status=status,
                message=message,
                duration=0,
                timestamp=datetime.utcnow(),
                details={
                    "disk_percent": percent,
                    "disk_free": disk.free,
                    "disk_total": disk.total
                }
            )
        
        except ImportError:
            return HealthResult(
                name="system_disk",
                status=HealthStatus.UNKNOWN,
                message="psutil not available",
                duration=0,
                timestamp=datetime.utcnow()
            )
    
    async def _check_network_connectivity(self) -> HealthResult:
        """Check network connectivity"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get('https://httpbin.org/status/200') as response:
                    if response.status == 200:
                        return HealthResult(
                            name="network_connectivity",
                            status=HealthStatus.HEALTHY,
                            message="Network connectivity OK",
                            duration=0,
                            timestamp=datetime.utcnow()
                        )
                    else:
                        return HealthResult(
                            name="network_connectivity",
                            status=HealthStatus.UNHEALTHY,
                            message=f"Network test failed: HTTP {response.status}",
                            duration=0,
                            timestamp=datetime.utcnow()
                        )
        
        except Exception as e:
            return HealthResult(
                name="network_connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"Network connectivity failed: {str(e)}",
                duration=0,
                timestamp=datetime.utcnow(),
                error=type(e).__name__
            )
    
    async def _check_redis_connection(self) -> HealthResult:
        """Check Redis connection"""
        try:
            redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                socket_timeout=3
            )
            
            redis_client.ping()
            
            return HealthResult(
                name="redis_connection",
                status=HealthStatus.HEALTHY,
                message="Redis connection OK",
                duration=0,
                timestamp=datetime.utcnow()
            )
        
        except Exception as e:
            return HealthResult(
                name="redis_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}",
                duration=0,
                timestamp=datetime.utcnow(),
                error=type(e).__name__
            )
    
    async def _check_postgres_connection(self) -> HealthResult:
        """Check PostgreSQL connection"""
        try:
            conn = psycopg2.connect(
                host=self.config.get('postgres_host', 'localhost'),
                port=self.config.get('postgres_port', 5432),
                database=self.config.get('postgres_db', 'nis_test'),
                user=self.config.get('postgres_user', 'test_user'),
                password=self.config.get('postgres_password', 'test_password'),
                connect_timeout=3
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            
            return HealthResult(
                name="postgres_connection",
                status=HealthStatus.HEALTHY,
                message="PostgreSQL connection OK",
                duration=0,
                timestamp=datetime.utcnow()
            )
        
        except Exception as e:
            return HealthResult(
                name="postgres_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"PostgreSQL connection failed: {str(e)}",
                duration=0,
                timestamp=datetime.utcnow(),
                error=type(e).__name__
            )
    
    async def _check_kafka_connection(self) -> HealthResult:
        """Check Kafka connection"""
        try:
            producer = KafkaProducer(
                bootstrap_servers=[f"{self.config.get('kafka_host', 'localhost')}:{self.config.get('kafka_port', 9092)}"],
                request_timeout_ms=3000
            )
            
            # Get cluster metadata
            metadata = producer.bootstrap_connected()
            producer.close()
            
            return HealthResult(
                name="kafka_connection",
                status=HealthStatus.HEALTHY,
                message="Kafka connection OK",
                duration=0,
                timestamp=datetime.utcnow()
            )
        
        except Exception as e:
            return HealthResult(
                name="kafka_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Kafka connection failed: {str(e)}",
                duration=0,
                timestamp=datetime.utcnow(),
                error=type(e).__name__
            )
    
    async def _check_agent_registry(self) -> HealthResult:
        """Check agent registry health"""
        try:
            # This would check your actual agent registry
            # For now, simulate the check
            agent_count = 5  # Mock agent count
            
            if agent_count > 0:
                return HealthResult(
                    name="agent_registry",
                    status=HealthStatus.HEALTHY,
                    message=f"Agent registry OK: {agent_count} agents registered",
                    duration=0,
                    timestamp=datetime.utcnow(),
                    details={"agent_count": agent_count}
                )
            else:
                return HealthResult(
                    name="agent_registry",
                    status=HealthStatus.DEGRADED,
                    message="No agents registered",
                    duration=0,
                    timestamp=datetime.utcnow()
                )
        
        except Exception as e:
            return HealthResult(
                name="agent_registry",
                status=HealthStatus.UNHEALTHY,
                message=f"Agent registry check failed: {str(e)}",
                duration=0,
                timestamp=datetime.utcnow(),
                error=type(e).__name__
            )
    
    async def _check_model_availability(self) -> HealthResult:
        """Check ML model availability"""
        try:
            # This would check your actual models
            # For now, simulate the check
            models_available = ["consciousness_v1", "reasoning_v2", "memory_v1"]
            
            if models_available:
                return HealthResult(
                    name="model_availability",
                    status=HealthStatus.HEALTHY,
                    message=f"Models available: {len(models_available)}",
                    duration=0,
                    timestamp=datetime.utcnow(),
                    details={"models": models_available}
                )
            else:
                return HealthResult(
                    name="model_availability",
                    status=HealthStatus.UNHEALTHY,
                    message="No models available",
                    duration=0,
                    timestamp=datetime.utcnow()
                )
        
        except Exception as e:
            return HealthResult(
                name="model_availability",
                status=HealthStatus.UNHEALTHY,
                message=f"Model availability check failed: {str(e)}",
                duration=0,
                timestamp=datetime.utcnow(),
                error=type(e).__name__
            )


# Global health monitor instance
_global_monitor: Optional[NISHealthMonitor] = None

def get_health_monitor(config: Dict[str, Any] = None) -> NISHealthMonitor:
    """Get global health monitor instance"""
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = NISHealthMonitor(config)
    
    return _global_monitor

async def start_health_monitoring(config: Dict[str, Any] = None):
    """Start health monitoring"""
    monitor = get_health_monitor(config)
    await monitor.start_monitoring()

async def stop_health_monitoring():
    """Stop health monitoring"""
    monitor = get_health_monitor()
    await monitor.stop_monitoring()

def get_system_health() -> SystemHealth:
    """Get current system health"""
    monitor = get_health_monitor()
    return monitor.get_system_health()


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'redis_host': 'localhost',
            'redis_port': 6380,
            'postgres_host': 'localhost',
            'postgres_port': 5433,
            'kafka_host': 'localhost',
            'kafka_port': 9093
        }
        
        monitor = NISHealthMonitor(config)
        
        print("Starting health monitoring...")
        await monitor.start_monitoring()
        
        # Let it run for a bit
        await asyncio.sleep(10)
        
        # Get health status
        health = monitor.get_system_health()
        print(f"\nSystem Health: {health.status.value}")
        print(f"Uptime: {health.uptime:.1f}s")
        print(f"Summary: {health.summary}")
        
        print("\nDetailed Results:")
        for name, result in health.checks.items():
            print(f"  {name}: {result.status.value} - {result.message}")
        
        # Print as JSON
        print("\nHealth JSON:")
        print(monitor.get_health_json())
        
        await monitor.stop_monitoring()
    
    asyncio.run(main())
