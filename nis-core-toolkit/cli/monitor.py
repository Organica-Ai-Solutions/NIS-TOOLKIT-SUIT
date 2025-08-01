#!/usr/bin/env python3
"""
NIS Core Toolkit - Real-Time Monitoring System
Comprehensive monitoring for consciousness, agent performance, and system health
"""

import asyncio
import json
import time
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
import threading
from collections import deque, defaultdict
from enum import Enum

# Optional imports with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import websockets
    import json
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    """Monitoring depth levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    load_average: List[float]
    uptime: float

@dataclass
class AgentMetrics:
    """Individual agent performance metrics"""
    agent_id: str
    agent_type: str
    timestamp: datetime
    response_time: float
    success_rate: float
    error_count: int
    consciousness_score: float
    kan_accuracy: float
    bias_detections: int
    coordination_efficiency: float
    memory_usage: float
    cpu_usage: float
    active_tasks: int

@dataclass
class ConsciousnessMetrics:
    """Consciousness-specific monitoring metrics"""
    agent_id: str
    timestamp: datetime
    self_awareness_score: float
    bias_detection_active: bool
    meta_cognitive_insights: List[str]
    attention_focus: List[str]
    ethical_violations: int
    uncertainty_acknowledgments: int
    reflection_depth: float
    consciousness_evolution: List[float]

@dataclass
class CoordinationMetrics:
    """Multi-agent coordination metrics"""
    timestamp: datetime
    active_agents: int
    coordination_requests: int
    coordination_successes: int
    coordination_failures: int
    average_response_time: float
    message_throughput: float
    consensus_accuracy: float
    network_efficiency: float

@dataclass
class Alert:
    """System alert with context"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    source: str
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class MonitoringConfig:
    """Monitoring system configuration"""
    monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD
    collection_interval: float = 5.0  # seconds
    retention_days: int = 30
    enable_consciousness_tracking: bool = True
    enable_real_time_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "error_rate": 5.0,
        "consciousness_degradation": 0.2,
        "coordination_failure_rate": 10.0
    })
    dashboard_port: int = 3000
    websocket_port: int = 3001
    database_path: str = "nis_monitoring.db"

class ConsciousnessMonitor:
    """
    Real-time consciousness monitoring for NIS agents
    
    Features:
    - Real-time consciousness state tracking
    - Bias detection monitoring
    - Meta-cognitive insight analysis
    - Consciousness evolution tracking
    - Ethical violation detection
    - Attention pattern analysis
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.consciousness_history = defaultdict(deque)
        self.bias_patterns = defaultdict(list)
        self.ethical_violations = defaultdict(list)
        self.attention_patterns = defaultdict(list)
        self.meta_insights = defaultdict(list)
        self.running = False
        
        self.logger = logging.getLogger(f"{__name__}.ConsciousnessMonitor")
        
    async def track_consciousness_state(self, agent_id: str, consciousness_data: Dict[str, Any]):
        """Track consciousness state for an agent"""
        
        metrics = ConsciousnessMetrics(
            agent_id=agent_id,
            timestamp=datetime.now(),
            self_awareness_score=consciousness_data.get("self_awareness_score", 0.0),
            bias_detection_active=consciousness_data.get("bias_detection_active", False),
            meta_cognitive_insights=consciousness_data.get("meta_cognitive_insights", []),
            attention_focus=consciousness_data.get("attention_focus", []),
            ethical_violations=consciousness_data.get("ethical_violations", 0),
            uncertainty_acknowledgments=consciousness_data.get("uncertainty_acknowledgments", 0),
            reflection_depth=consciousness_data.get("reflection_depth", 0.0),
            consciousness_evolution=consciousness_data.get("consciousness_evolution", [])
        )
        
        # Store in history with retention
        self.consciousness_history[agent_id].append(metrics)
        max_history = int((self.config.retention_days * 24 * 3600) / self.config.collection_interval)
        if len(self.consciousness_history[agent_id]) > max_history:
            self.consciousness_history[agent_id].popleft()
        
        # Analyze consciousness patterns
        await self._analyze_consciousness_patterns(agent_id, metrics)
        
        # Check for consciousness degradation
        await self._check_consciousness_health(agent_id, metrics)
        
        self.logger.debug(f"Tracked consciousness for {agent_id}: awareness={metrics.self_awareness_score:.3f}")
    
    async def _analyze_consciousness_patterns(self, agent_id: str, metrics: ConsciousnessMetrics):
        """Analyze consciousness patterns for trends"""
        
        history = list(self.consciousness_history[agent_id])
        if len(history) < 5:  # Need minimum history for pattern analysis
            return
        
        # Analyze consciousness evolution trend
        recent_scores = [m.self_awareness_score for m in history[-10:]]
        if len(recent_scores) >= 5:
            trend = statistics.mean(recent_scores[-3:]) - statistics.mean(recent_scores[:3])
            
            if trend < -0.1:  # Significant degradation
                await self._trigger_consciousness_alert(
                    agent_id, 
                    AlertSeverity.WARNING,
                    f"Consciousness degradation detected: {trend:.3f} drop in awareness"
                )
        
        # Track bias patterns
        if metrics.bias_detection_active and len(metrics.meta_cognitive_insights) > 0:
            bias_insights = [insight for insight in metrics.meta_cognitive_insights if "bias" in insight.lower()]
            if bias_insights:
                self.bias_patterns[agent_id].extend(bias_insights)
        
        # Monitor ethical violations
        if metrics.ethical_violations > 0:
            self.ethical_violations[agent_id].append({
                "timestamp": metrics.timestamp,
                "violations": metrics.ethical_violations,
                "context": metrics.meta_cognitive_insights
            })
    
    async def _check_consciousness_health(self, agent_id: str, metrics: ConsciousnessMetrics):
        """Check consciousness health and trigger alerts if needed"""
        
        # Check consciousness score threshold
        if metrics.self_awareness_score < 0.5:
            await self._trigger_consciousness_alert(
                agent_id,
                AlertSeverity.ERROR,
                f"Low consciousness score: {metrics.self_awareness_score:.3f}"
            )
        
        # Check for bias detection failures
        if not metrics.bias_detection_active:
            await self._trigger_consciousness_alert(
                agent_id,
                AlertSeverity.WARNING,
                "Bias detection system inactive"
            )
        
        # Check for ethical violations
        if metrics.ethical_violations > 0:
            await self._trigger_consciousness_alert(
                agent_id,
                AlertSeverity.CRITICAL,
                f"Ethical violations detected: {metrics.ethical_violations}"
            )
    
    async def _trigger_consciousness_alert(self, agent_id: str, severity: AlertSeverity, message: str):
        """Trigger consciousness-related alert"""
        
        alert = Alert(
            alert_id=f"consciousness_{agent_id}_{int(time.time())}",
            timestamp=datetime.now(),
            severity=severity,
            source=f"consciousness_monitor_{agent_id}",
            message=message,
            details={
                "agent_id": agent_id,
                "consciousness_history": list(self.consciousness_history[agent_id])[-5:],
                "bias_patterns": self.bias_patterns[agent_id][-5:],
                "ethical_violations": self.ethical_violations[agent_id][-3:]
            }
        )
        
        # This would be sent to the main monitoring system
        self.logger.warning(f"Consciousness Alert [{severity.value}]: {message} (Agent: {agent_id})")
    
    def get_consciousness_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get consciousness monitoring summary for an agent"""
        
        if agent_id not in self.consciousness_history:
            return {"agent_id": agent_id, "status": "no_data"}
        
        history = list(self.consciousness_history[agent_id])
        if not history:
            return {"agent_id": agent_id, "status": "no_data"}
        
        latest = history[-1]
        recent_scores = [m.self_awareness_score for m in history[-10:]]
        
        return {
            "agent_id": agent_id,
            "current_consciousness_score": latest.self_awareness_score,
            "consciousness_trend": statistics.mean(recent_scores[-3:]) - statistics.mean(recent_scores[:3]) if len(recent_scores) >= 6 else 0.0,
            "bias_detection_active": latest.bias_detection_active,
            "recent_insights": latest.meta_cognitive_insights[-3:],
            "attention_areas": len(latest.attention_focus),
            "ethical_violations_total": sum(v["violations"] for v in self.ethical_violations[agent_id]),
            "consciousness_stability": statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0.0,
            "last_update": latest.timestamp.isoformat()
        }

class PerformanceMonitor:
    """
    Performance monitoring for NIS agents and system components
    
    Features:
    - Real-time performance tracking
    - Resource usage monitoring
    - Response time analysis
    - Success rate tracking
    - Bottleneck detection
    - Capacity planning metrics
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.agent_metrics_history = defaultdict(deque)
        self.system_metrics_history = deque()
        self.performance_baselines = {}
        self.running = False
        
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
    
    async def track_agent_performance(self, agent_id: str, performance_data: Dict[str, Any]):
        """Track performance metrics for an agent"""
        
        metrics = AgentMetrics(
            agent_id=agent_id,
            agent_type=performance_data.get("agent_type", "unknown"),
            timestamp=datetime.now(),
            response_time=performance_data.get("response_time", 0.0),
            success_rate=performance_data.get("success_rate", 0.0),
            error_count=performance_data.get("error_count", 0),
            consciousness_score=performance_data.get("consciousness_score", 0.0),
            kan_accuracy=performance_data.get("kan_accuracy", 0.0),
            bias_detections=performance_data.get("bias_detections", 0),
            coordination_efficiency=performance_data.get("coordination_efficiency", 0.0),
            memory_usage=performance_data.get("memory_usage", 0.0),
            cpu_usage=performance_data.get("cpu_usage", 0.0),
            active_tasks=performance_data.get("active_tasks", 0)
        )
        
        # Store in history
        self.agent_metrics_history[agent_id].append(metrics)
        max_history = int((self.config.retention_days * 24 * 3600) / self.config.collection_interval)
        if len(self.agent_metrics_history[agent_id]) > max_history:
            self.agent_metrics_history[agent_id].popleft()
        
        # Analyze performance patterns
        await self._analyze_performance_patterns(agent_id, metrics)
        
        # Check performance thresholds
        await self._check_performance_thresholds(agent_id, metrics)
        
        self.logger.debug(f"Tracked performance for {agent_id}: response_time={metrics.response_time:.3f}ms")
    
    async def track_system_performance(self):
        """Track system-level performance metrics"""
        
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available, system metrics limited")
            return
        
        try:
            # Collect system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                process_count=len(psutil.pids()),
                load_average=psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0],
                uptime=time.time() - psutil.boot_time()
            )
            
            # Store in history
            self.system_metrics_history.append(metrics)
            max_history = int((self.config.retention_days * 24 * 3600) / self.config.collection_interval)
            if len(self.system_metrics_history) > max_history:
                self.system_metrics_history.popleft()
            
            # Check system thresholds
            await self._check_system_thresholds(metrics)
            
            self.logger.debug(f"System metrics: CPU={cpu_usage:.1f}%, Memory={memory.percent:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _analyze_performance_patterns(self, agent_id: str, metrics: AgentMetrics):
        """Analyze performance patterns and trends"""
        
        history = list(self.agent_metrics_history[agent_id])
        if len(history) < 10:
            return
        
        # Analyze response time trends
        recent_times = [m.response_time for m in history[-10:]]
        if statistics.mean(recent_times) > statistics.mean([m.response_time for m in history[-20:-10]]) * 1.5:
            await self._trigger_performance_alert(
                agent_id,
                AlertSeverity.WARNING,
                f"Response time degradation detected: {statistics.mean(recent_times):.3f}ms average"
            )
        
        # Analyze success rate trends
        recent_success = [m.success_rate for m in history[-5:]]
        if statistics.mean(recent_success) < 0.9:
            await self._trigger_performance_alert(
                agent_id,
                AlertSeverity.ERROR,
                f"Low success rate: {statistics.mean(recent_success):.1%}"
            )
    
    async def _check_performance_thresholds(self, agent_id: str, metrics: AgentMetrics):
        """Check performance against configured thresholds"""
        
        # Check response time
        if metrics.response_time > 1000:  # 1 second
            await self._trigger_performance_alert(
                agent_id,
                AlertSeverity.WARNING,
                f"High response time: {metrics.response_time:.3f}ms"
            )
        
        # Check success rate
        if metrics.success_rate < 0.95:
            await self._trigger_performance_alert(
                agent_id,
                AlertSeverity.ERROR,
                f"Low success rate: {metrics.success_rate:.1%}"
            )
        
        # Check consciousness score
        if metrics.consciousness_score < 0.7:
            await self._trigger_performance_alert(
                agent_id,
                AlertSeverity.WARNING,
                f"Low consciousness performance: {metrics.consciousness_score:.3f}"
            )
    
    async def _check_system_thresholds(self, metrics: SystemMetrics):
        """Check system metrics against thresholds"""
        
        thresholds = self.config.alert_thresholds
        
        if metrics.cpu_usage > thresholds.get("cpu_usage", 80.0):
            await self._trigger_system_alert(
                AlertSeverity.WARNING,
                f"High CPU usage: {metrics.cpu_usage:.1f}%"
            )
        
        if metrics.memory_usage > thresholds.get("memory_usage", 85.0):
            await self._trigger_system_alert(
                AlertSeverity.ERROR,
                f"High memory usage: {metrics.memory_usage:.1f}%"
            )
    
    async def _trigger_performance_alert(self, agent_id: str, severity: AlertSeverity, message: str):
        """Trigger performance-related alert"""
        self.logger.warning(f"Performance Alert [{severity.value}]: {message} (Agent: {agent_id})")
    
    async def _trigger_system_alert(self, severity: AlertSeverity, message: str):
        """Trigger system-level alert"""
        self.logger.warning(f"System Alert [{severity.value}]: {message}")
    
    def get_performance_summary(self, agent_id: str = None) -> Dict[str, Any]:
        """Get performance summary for agent or entire system"""
        
        if agent_id:
            # Agent-specific summary
            if agent_id not in self.agent_metrics_history:
                return {"agent_id": agent_id, "status": "no_data"}
            
            history = list(self.agent_metrics_history[agent_id])
            if not history:
                return {"agent_id": agent_id, "status": "no_data"}
            
            latest = history[-1]
            recent_metrics = history[-10:] if len(history) >= 10 else history
            
            return {
                "agent_id": agent_id,
                "current_response_time": latest.response_time,
                "avg_response_time": statistics.mean([m.response_time for m in recent_metrics]),
                "success_rate": latest.success_rate,
                "error_count": latest.error_count,
                "consciousness_score": latest.consciousness_score,
                "kan_accuracy": latest.kan_accuracy,
                "coordination_efficiency": latest.coordination_efficiency,
                "resource_usage": {
                    "memory": latest.memory_usage,
                    "cpu": latest.cpu_usage
                },
                "active_tasks": latest.active_tasks,
                "last_update": latest.timestamp.isoformat()
            }
        else:
            # System-wide summary
            if not self.system_metrics_history:
                return {"status": "no_system_data"}
            
            latest_system = self.system_metrics_history[-1]
            recent_system = list(self.system_metrics_history)[-10:] if len(self.system_metrics_history) >= 10 else list(self.system_metrics_history)
            
            # Aggregate agent metrics
            all_agent_metrics = []
            for agent_history in self.agent_metrics_history.values():
                if agent_history:
                    all_agent_metrics.append(agent_history[-1])
            
            return {
                "system_metrics": {
                    "cpu_usage": latest_system.cpu_usage,
                    "memory_usage": latest_system.memory_usage,
                    "disk_usage": latest_system.disk_usage,
                    "process_count": latest_system.process_count,
                    "uptime": latest_system.uptime
                },
                "agent_summary": {
                    "total_agents": len(all_agent_metrics),
                    "avg_response_time": statistics.mean([m.response_time for m in all_agent_metrics]) if all_agent_metrics else 0,
                    "avg_success_rate": statistics.mean([m.success_rate for m in all_agent_metrics]) if all_agent_metrics else 0,
                    "total_errors": sum([m.error_count for m in all_agent_metrics]),
                    "avg_consciousness": statistics.mean([m.consciousness_score for m in all_agent_metrics]) if all_agent_metrics else 0
                },
                "last_update": latest_system.timestamp.isoformat()
            }

class CoordinationMonitor:
    """
    Multi-agent coordination monitoring
    
    Features:
    - Coordination request tracking
    - Success/failure rate monitoring
    - Message throughput analysis
    - Consensus accuracy tracking
    - Network efficiency metrics
    - Bottleneck identification
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.coordination_history = deque()
        self.message_patterns = defaultdict(list)
        self.consensus_tracking = []
        self.network_topology = {}
        
        self.logger = logging.getLogger(f"{__name__}.CoordinationMonitor")
    
    async def track_coordination_event(self, coordination_data: Dict[str, Any]):
        """Track multi-agent coordination event"""
        
        metrics = CoordinationMetrics(
            timestamp=datetime.now(),
            active_agents=coordination_data.get("active_agents", 0),
            coordination_requests=coordination_data.get("coordination_requests", 0),
            coordination_successes=coordination_data.get("coordination_successes", 0),
            coordination_failures=coordination_data.get("coordination_failures", 0),
            average_response_time=coordination_data.get("average_response_time", 0.0),
            message_throughput=coordination_data.get("message_throughput", 0.0),
            consensus_accuracy=coordination_data.get("consensus_accuracy", 0.0),
            network_efficiency=coordination_data.get("network_efficiency", 0.0)
        )
        
        # Store in history
        self.coordination_history.append(metrics)
        max_history = int((self.config.retention_days * 24 * 3600) / self.config.collection_interval)
        if len(self.coordination_history) > max_history:
            self.coordination_history.popleft()
        
        # Analyze coordination patterns
        await self._analyze_coordination_patterns(metrics)
        
        self.logger.debug(f"Tracked coordination: {metrics.active_agents} agents, {metrics.coordination_successes}/{metrics.coordination_requests} success rate")
    
    async def _analyze_coordination_patterns(self, metrics: CoordinationMetrics):
        """Analyze coordination patterns for optimization opportunities"""
        
        history = list(self.coordination_history)
        if len(history) < 5:
            return
        
        # Check coordination success rate
        recent_metrics = history[-5:]
        total_requests = sum([m.coordination_requests for m in recent_metrics])
        total_successes = sum([m.coordination_successes for m in recent_metrics])
        
        if total_requests > 0:
            success_rate = total_successes / total_requests
            if success_rate < 0.9:
                await self._trigger_coordination_alert(
                    AlertSeverity.WARNING,
                    f"Low coordination success rate: {success_rate:.1%}"
                )
        
        # Check network efficiency trends
        recent_efficiency = [m.network_efficiency for m in recent_metrics]
        if statistics.mean(recent_efficiency) < 0.7:
            await self._trigger_coordination_alert(
                AlertSeverity.WARNING,
                f"Low network efficiency: {statistics.mean(recent_efficiency):.1%}"
            )
    
    async def _trigger_coordination_alert(self, severity: AlertSeverity, message: str):
        """Trigger coordination-related alert"""
        self.logger.warning(f"Coordination Alert [{severity.value}]: {message}")
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get coordination monitoring summary"""
        
        if not self.coordination_history:
            return {"status": "no_data"}
        
        history = list(self.coordination_history)
        latest = history[-1]
        recent_metrics = history[-10:] if len(history) >= 10 else history
        
        total_requests = sum([m.coordination_requests for m in recent_metrics])
        total_successes = sum([m.coordination_successes for m in recent_metrics])
        
        return {
            "current_active_agents": latest.active_agents,
            "recent_success_rate": (total_successes / total_requests) if total_requests > 0 else 0.0,
            "average_response_time": statistics.mean([m.average_response_time for m in recent_metrics]),
            "message_throughput": latest.message_throughput,
            "consensus_accuracy": latest.consensus_accuracy,
            "network_efficiency": latest.network_efficiency,
            "coordination_trends": {
                "requests_trend": total_requests,
                "efficiency_trend": statistics.mean([m.network_efficiency for m in recent_metrics])
            },
            "last_update": latest.timestamp.isoformat()
        }

class NISMonitoringSystem:
    """
    Comprehensive NIS monitoring system that integrates all monitoring components
    
    Features:
    - Real-time consciousness tracking
    - Performance monitoring
    - Multi-agent coordination monitoring
    - System health monitoring
    - Alert management
    - Real-time dashboard
    - WebSocket API for live updates
    """
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        
        # Initialize monitoring components
        self.consciousness_monitor = ConsciousnessMonitor(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.coordination_monitor = CoordinationMonitor(self.config)
        
        # Alert management
        self.alerts = deque()
        self.alert_handlers = []
        self.subscribers = set()
        
        # System state
        self.running = False
        self.start_time = None
        
        # Background tasks
        self.monitoring_tasks = []
        
        self.logger = logging.getLogger(f"{__name__}.NISMonitoringSystem")
        
        # Initialize database if available
        if SQLITE_AVAILABLE:
            self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect(self.config.database_path)
            cursor = conn.cursor()
            
            # Create tables for different metric types
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS consciousness_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    timestamp TEXT,
                    self_awareness_score REAL,
                    bias_detection_active BOOLEAN,
                    ethical_violations INTEGER,
                    meta_insights TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    timestamp TEXT,
                    response_time REAL,
                    success_rate REAL,
                    error_count INTEGER,
                    consciousness_score REAL,
                    kan_accuracy REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    process_count INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE,
                    timestamp TEXT,
                    severity TEXT,
                    source TEXT,
                    message TEXT,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.running:
            self.logger.warning("Monitoring system already running")
            return
        
        self.running = True
        self.start_time = datetime.now()
        
        self.logger.info("Starting NIS monitoring system")
        
        # Start background monitoring tasks
        if self.config.monitoring_level in [MonitoringLevel.STANDARD, MonitoringLevel.COMPREHENSIVE, MonitoringLevel.DEEP]:
            self.monitoring_tasks.extend([
                asyncio.create_task(self._system_monitoring_loop()),
                asyncio.create_task(self._alert_processing_loop()),
                asyncio.create_task(self._cleanup_old_data_loop())
            ])
        
        # Start WebSocket server for real-time updates
        if WEBSOCKETS_AVAILABLE and self.config.enable_real_time_alerts:
            self.monitoring_tasks.append(
                asyncio.create_task(self._start_websocket_server())
            )
        
        self.logger.info(f"Monitoring system started with {len(self.monitoring_tasks)} background tasks")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all background tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        self.logger.info("Monitoring system stopped")
    
    async def _system_monitoring_loop(self):
        """Background loop for system-level monitoring"""
        while self.running:
            try:
                await self.performance_monitor.track_system_performance()
                await asyncio.sleep(self.config.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self.config.collection_interval)
    
    async def _alert_processing_loop(self):
        """Background loop for processing alerts"""
        while self.running:
            try:
                # Process queued alerts
                if self.alerts:
                    alert = self.alerts.popleft()
                    await self._process_alert(alert)
                
                await asyncio.sleep(1.0)  # Check for alerts every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _cleanup_old_data_loop(self):
        """Background loop for cleaning up old monitoring data"""
        while self.running:
            try:
                # Clean up old data every hour
                await asyncio.sleep(3600)
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Data cleanup error: {e}")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time monitoring updates"""
        try:
            async def handle_websocket(websocket, path):
                self.subscribers.add(websocket)
                try:
                    await websocket.wait_closed()
                finally:
                    self.subscribers.discard(websocket)
            
            server = await websockets.serve(
                handle_websocket,
                "localhost",
                self.config.websocket_port
            )
            
            self.logger.info(f"WebSocket server started on port {self.config.websocket_port}")
            await server.wait_closed()
            
        except Exception as e:
            self.logger.error(f"WebSocket server error: {e}")
    
    async def _process_alert(self, alert: Alert):
        """Process and distribute alert"""
        
        # Store alert in database
        if SQLITE_AVAILABLE:
            try:
                conn = sqlite3.connect(self.config.database_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts (alert_id, timestamp, severity, source, message, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id,
                    alert.timestamp.isoformat(),
                    alert.severity.value,
                    alert.source,
                    alert.message,
                    json.dumps(alert.details)
                ))
                conn.commit()
                conn.close()
            except Exception as e:
                self.logger.error(f"Failed to store alert in database: {e}")
        
        # Send to WebSocket subscribers
        if self.subscribers:
            alert_message = {
                "type": "alert",
                "data": {
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity.value,
                    "source": alert.source,
                    "message": alert.message
                }
            }
            
            disconnected = set()
            for websocket in self.subscribers:
                try:
                    await websocket.send(json.dumps(alert_message))
                except:
                    disconnected.add(websocket)
            
            # Remove disconnected subscribers
            self.subscribers -= disconnected
        
        # Call custom alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
        
        if SQLITE_AVAILABLE:
            try:
                conn = sqlite3.connect(self.config.database_path)
                cursor = conn.cursor()
                
                # Clean up old records
                cursor.execute('DELETE FROM consciousness_metrics WHERE timestamp < ?', (cutoff_time.isoformat(),))
                cursor.execute('DELETE FROM performance_metrics WHERE timestamp < ?', (cutoff_time.isoformat(),))
                cursor.execute('DELETE FROM system_metrics WHERE timestamp < ?', (cutoff_time.isoformat(),))
                cursor.execute('DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE', (cutoff_time.isoformat(),))
                
                conn.commit()
                conn.close()
                
                self.logger.debug("Old monitoring data cleaned up")
                
            except Exception as e:
                self.logger.error(f"Data cleanup failed: {e}")
    
    # Public API methods
    
    async def track_agent_consciousness(self, agent_id: str, consciousness_data: Dict[str, Any]):
        """Track consciousness metrics for an agent"""
        await self.consciousness_monitor.track_consciousness_state(agent_id, consciousness_data)
    
    async def track_agent_performance(self, agent_id: str, performance_data: Dict[str, Any]):
        """Track performance metrics for an agent"""
        await self.performance_monitor.track_agent_performance(agent_id, performance_data)
    
    async def track_coordination_event(self, coordination_data: Dict[str, Any]):
        """Track multi-agent coordination event"""
        await self.coordination_monitor.track_coordination_event(coordination_data)
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "monitoring_system": {
                "running": self.running,
                "uptime_seconds": uptime,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "monitoring_level": self.config.monitoring_level.value,
                "collection_interval": self.config.collection_interval,
                "active_tasks": len(self.monitoring_tasks),
                "websocket_subscribers": len(self.subscribers)
            },
            "consciousness_monitoring": {
                "tracked_agents": len(self.consciousness_monitor.consciousness_history),
                "total_consciousness_records": sum(len(history) for history in self.consciousness_monitor.consciousness_history.values()),
                "bias_patterns_detected": sum(len(patterns) for patterns in self.consciousness_monitor.bias_patterns.values()),
                "ethical_violations": sum(len(violations) for violations in self.consciousness_monitor.ethical_violations.values())
            },
            "performance_monitoring": {
                "tracked_agents": len(self.performance_monitor.agent_metrics_history),
                "system_metrics_available": len(self.performance_monitor.system_metrics_history) > 0,
                "total_performance_records": sum(len(history) for history in self.performance_monitor.agent_metrics_history.values())
            },
            "coordination_monitoring": {
                "coordination_events": len(self.coordination_monitor.coordination_history),
                "recent_success_rate": self.coordination_monitor.get_coordination_summary().get("recent_success_rate", 0.0)
            },
            "alerts": {
                "total_alerts": len(self.alerts),
                "unresolved_alerts": len([a for a in self.alerts if not a.resolved]),
                "alert_handlers": len(self.alert_handlers)
            }
        }
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a specific agent"""
        
        consciousness_summary = self.consciousness_monitor.get_consciousness_summary(agent_id)
        performance_summary = self.performance_monitor.get_performance_summary(agent_id)
        
        return {
            "agent_id": agent_id,
            "consciousness": consciousness_summary,
            "performance": performance_summary,
            "last_seen": max(
                consciousness_summary.get("last_update", ""),
                performance_summary.get("last_update", "")
            ) if consciousness_summary.get("status") != "no_data" or performance_summary.get("status") != "no_data" else None
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        
        system_status = self.get_system_status()
        performance_summary = self.performance_monitor.get_performance_summary()
        coordination_summary = self.coordination_monitor.get_coordination_summary()
        
        # Get summaries for all tracked agents
        agent_summaries = {}
        for agent_id in self.consciousness_monitor.consciousness_history.keys():
            agent_summaries[agent_id] = self.get_agent_status(agent_id)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": system_status,
            "system_performance": performance_summary,
            "coordination_status": coordination_summary,
            "agents": agent_summaries,
            "recent_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity.value,
                    "source": alert.source,
                    "message": alert.message,
                    "resolved": alert.resolved
                }
                for alert in list(self.alerts)[-10:]  # Last 10 alerts
            ]
        }

# Factory function for creating monitoring system
def create_monitoring_system(config: Dict[str, Any] = None) -> NISMonitoringSystem:
    """Create NIS monitoring system with configuration"""
    
    if config:
        monitoring_config = MonitoringConfig(**config)
    else:
        monitoring_config = MonitoringConfig()
    
    return NISMonitoringSystem(monitoring_config)

# Example usage
async def example_monitoring_usage():
    """Example of how to use the NIS monitoring system"""
    
    # Create monitoring system
    monitoring = create_monitoring_system({
        "monitoring_level": "comprehensive",
        "collection_interval": 2.0,
        "enable_real_time_alerts": True,
        "dashboard_port": 3000
    })
    
    # Start monitoring
    await monitoring.start_monitoring()
    
    # Simulate tracking some agent metrics
    await monitoring.track_agent_consciousness("reasoning-agent-1", {
        "self_awareness_score": 0.87,
        "bias_detection_active": True,
        "meta_cognitive_insights": ["High confidence in reasoning", "No biases detected"],
        "attention_focus": ["problem_analysis", "solution_generation"],
        "ethical_violations": 0,
        "uncertainty_acknowledgments": 2
    })
    
    await monitoring.track_agent_performance("reasoning-agent-1", {
        "agent_type": "reasoning",
        "response_time": 45.2,
        "success_rate": 0.96,
        "error_count": 1,
        "consciousness_score": 0.87,
        "kan_accuracy": 0.94,
        "coordination_efficiency": 0.91,
        "memory_usage": 12.5,
        "cpu_usage": 8.3,
        "active_tasks": 3
    })
    
    # Get dashboard data
    dashboard_data = monitoring.get_dashboard_data()
    print(f"Dashboard data: {json.dumps(dashboard_data, indent=2)}")
    
    # Stop monitoring
    await monitoring.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(example_monitoring_usage()) 