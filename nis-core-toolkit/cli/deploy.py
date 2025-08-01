#!/usr/bin/env python3
"""
NIS Core Toolkit - Deployment System
Automated deployment for NIS-powered systems with multi-platform support
"""

import asyncio
import json
import os
import sys
import yaml
import subprocess
import tempfile
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import hashlib
import secrets

# Optional imports with graceful fallback
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    import kubernetes
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentPlatform(Enum):
    """Supported deployment platforms"""
    LOCAL = "local"
    DOCKER = "docker"
    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES = "kubernetes"
    AWS_ECS = "aws_ecs"
    AWS_LAMBDA = "aws_lambda"
    GCP_CLOUD_RUN = "gcp_cloud_run"
    AZURE_CONTAINER = "azure_container"
    HEROKU = "heroku"
    VERCEL = "vercel"
    RAILWAY = "railway"

class DeploymentStatus(Enum):
    """Deployment status states"""
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"
    UPDATING = "updating"
    ROLLING_BACK = "rolling_back"

class HealthCheckType(Enum):
    """Health check types"""
    HTTP = "http"
    TCP = "tcp"
    COMMAND = "command"
    CONSCIOUSNESS = "consciousness"
    AGENT_STATUS = "agent_status"

@dataclass
class HealthCheck:
    """Health check configuration"""
    type: HealthCheckType
    endpoint: Optional[str] = None
    port: Optional[int] = None
    command: Optional[str] = None
    interval: int = 30  # seconds
    timeout: int = 5  # seconds
    retries: int = 3
    initial_delay: int = 10  # seconds

@dataclass
class ScalingConfig:
    """Auto-scaling configuration"""
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70  # percentage
    target_memory_utilization: int = 80  # percentage
    target_consciousness_load: float = 0.8  # consciousness-aware scaling
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_tracing: bool = False
    consciousness_monitoring: bool = True
    performance_monitoring: bool = True
    custom_metrics: List[str] = field(default_factory=list)
    dashboard_enabled: bool = True
    alert_webhooks: List[str] = field(default_factory=list)

@dataclass
class SecurityConfig:
    """Security configuration for deployment"""
    enable_tls: bool = True
    certificate_mode: str = "auto"  # auto, manual, letsencrypt
    network_policies: List[Dict[str, Any]] = field(default_factory=list)
    secrets_management: str = "auto"  # auto, vault, k8s-secrets
    authentication_required: bool = True
    cors_enabled: bool = True
    rate_limiting: bool = True

@dataclass
class DeploymentConfig:
    """Comprehensive deployment configuration"""
    name: str
    platform: DeploymentPlatform
    image: Optional[str] = None
    build_context: str = "."
    dockerfile: str = "Dockerfile"
    port: int = 8000
    replicas: int = 1
    environment: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=lambda: {
        "cpu": "500m",
        "memory": "512Mi",
        "gpu": None
    })
    volumes: List[Dict[str, str]] = field(default_factory=list)
    health_checks: List[HealthCheck] = field(default_factory=list)
    scaling: Optional[ScalingConfig] = None
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    custom_annotations: Dict[str, str] = field(default_factory=dict)
    rollback_strategy: str = "previous_version"  # previous_version, canary, blue_green

@dataclass
class DeploymentResult:
    """Deployment result with comprehensive information"""
    deployment_id: str
    status: DeploymentStatus
    platform: DeploymentPlatform
    start_time: datetime
    end_time: Optional[datetime] = None
    urls: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    rollback_info: Optional[Dict[str, Any]] = None
    health_status: Dict[str, Any] = field(default_factory=dict)

class DockerDeployer:
    """Docker deployment handler"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.docker_client = None
        self.logger = logging.getLogger(f"{__name__}.DockerDeployer")
        
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                self.logger.warning(f"Docker client initialization failed: {e}")
    
    async def deploy(self) -> DeploymentResult:
        """Deploy to Docker"""
        
        result = DeploymentResult(
            deployment_id=f"docker_{self.config.name}_{int(datetime.now().timestamp())}",
            status=DeploymentStatus.PENDING,
            platform=DeploymentPlatform.DOCKER,
            start_time=datetime.now()
        )
        
        if not self.docker_client:
            result.status = DeploymentStatus.FAILED
            result.errors.append("Docker client not available")
            return result
        
        try:
            # Build Docker image
            result.status = DeploymentStatus.BUILDING
            image_tag = f"{self.config.name}:latest"
            
            self.logger.info(f"Building Docker image: {image_tag}")
            image, build_logs = self.docker_client.images.build(
                path=self.config.build_context,
                dockerfile=self.config.dockerfile,
                tag=image_tag,
                rm=True,
                pull=True
            )
            
            for log in build_logs:
                if 'stream' in log:
                    result.logs.append(log['stream'].strip())
            
            # Deploy container
            result.status = DeploymentStatus.DEPLOYING
            self.logger.info(f"Starting container: {self.config.name}")
            
            container_config = {
                "name": self.config.name,
                "image": image_tag,
                "ports": {f"{self.config.port}/tcp": self.config.port},
                "environment": self.config.environment,
                "detach": True,
                "restart_policy": {"Name": "unless-stopped"},
                "labels": {
                    "nis.deployment.id": result.deployment_id,
                    "nis.deployment.platform": "docker",
                    "nis.deployment.timestamp": result.start_time.isoformat()
                }
            }
            
            # Add volume mounts
            if self.config.volumes:
                container_config["volumes"] = {
                    vol["host_path"]: {"bind": vol["container_path"], "mode": vol.get("mode", "rw")}
                    for vol in self.config.volumes
                }
            
            # Stop existing container if it exists
            try:
                existing_container = self.docker_client.containers.get(self.config.name)
                existing_container.stop()
                existing_container.remove()
                self.logger.info(f"Stopped and removed existing container: {self.config.name}")
            except docker.errors.NotFound:
                pass  # Container doesn't exist, which is fine
            
            # Start new container
            container = self.docker_client.containers.run(**container_config)
            
            # Wait for container to be ready
            await self._wait_for_container_health(container, result)
            
            if result.status != DeploymentStatus.FAILED:
                result.status = DeploymentStatus.RUNNING
                result.urls = [f"http://localhost:{self.config.port}"]
                
                # Setup monitoring if enabled
                if self.config.monitoring.enable_metrics:
                    await self._setup_docker_monitoring(container, result)
            
            result.end_time = datetime.now()
            self.logger.info(f"Docker deployment completed: {result.deployment_id}")
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.errors.append(f"Docker deployment failed: {str(e)}")
            result.end_time = datetime.now()
            self.logger.error(f"Docker deployment error: {e}")
        
        return result
    
    async def _wait_for_container_health(self, container, result: DeploymentResult):
        """Wait for container to be healthy"""
        
        max_wait = 300  # 5 minutes
        wait_interval = 5
        elapsed = 0
        
        while elapsed < max_wait:
            try:
                container.reload()
                
                if container.status == "running":
                    # Check custom health checks
                    if self.config.health_checks:
                        health_ok = await self._check_container_health(container)
                        if health_ok:
                            return
                    else:
                        # No health checks defined, assume healthy if running
                        return
                
                elif container.status in ["exited", "dead"]:
                    result.status = DeploymentStatus.FAILED
                    result.errors.append(f"Container failed to start: {container.status}")
                    return
                
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval
                
            except Exception as e:
                result.errors.append(f"Health check error: {str(e)}")
                break
        
        # Timeout reached
        result.status = DeploymentStatus.FAILED
        result.errors.append("Container health check timeout")
    
    async def _check_container_health(self, container) -> bool:
        """Check container health using configured health checks"""
        
        for health_check in self.config.health_checks:
            if health_check.type == HealthCheckType.HTTP:
                # HTTP health check
                if REQUESTS_AVAILABLE:
                    try:
                        url = f"http://localhost:{self.config.port}{health_check.endpoint or '/health'}"
                        response = requests.get(url, timeout=health_check.timeout)
                        if response.status_code != 200:
                            return False
                    except:
                        return False
                        
            elif health_check.type == HealthCheckType.COMMAND:
                # Command health check
                try:
                    result = container.exec_run(health_check.command, timeout=health_check.timeout)
                    if result.exit_code != 0:
                        return False
                except:
                    return False
        
        return True
    
    async def _setup_docker_monitoring(self, container, result: DeploymentResult):
        """Setup monitoring for Docker container"""
        
        try:
            # Get container stats
            stats = container.stats(stream=False)
            
            result.metrics.update({
                "container_id": container.id[:12],
                "container_status": container.status,
                "cpu_usage": self._calculate_cpu_percentage(stats),
                "memory_usage": stats.get("memory_stats", {}).get("usage", 0),
                "memory_limit": stats.get("memory_stats", {}).get("limit", 0),
                "network_io": stats.get("networks", {}),
                "block_io": stats.get("blkio_stats", {})
            })
            
        except Exception as e:
            self.logger.warning(f"Failed to setup Docker monitoring: {e}")
    
    def _calculate_cpu_percentage(self, stats: Dict[str, Any]) -> float:
        """Calculate CPU usage percentage from Docker stats"""
        
        try:
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})
            
            cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - \
                       precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
            
            system_delta = cpu_stats.get("system_cpu_usage", 0) - \
                          precpu_stats.get("system_cpu_usage", 0)
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * len(cpu_stats.get("cpu_usage", {}).get("percpu_usage", [])) * 100
                return round(cpu_percent, 2)
        except:
            pass
        
        return 0.0

class KubernetesDeployer:
    """Kubernetes deployment handler"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.k8s_client = None
        self.apps_v1 = None
        self.core_v1 = None
        self.logger = logging.getLogger(f"{__name__}.KubernetesDeployer")
        
        if KUBERNETES_AVAILABLE:
            try:
                # Try to load kube config
                try:
                    config.load_kube_config()
                except:
                    # Try in-cluster config
                    config.load_incluster_config()
                
                self.apps_v1 = client.AppsV1Api()
                self.core_v1 = client.CoreV1Api()
                self.logger.info("Kubernetes client initialized")
                
            except Exception as e:
                self.logger.warning(f"Kubernetes client initialization failed: {e}")
    
    async def deploy(self) -> DeploymentResult:
        """Deploy to Kubernetes"""
        
        result = DeploymentResult(
            deployment_id=f"k8s_{self.config.name}_{int(datetime.now().timestamp())}",
            status=DeploymentStatus.PENDING,
            platform=DeploymentPlatform.KUBERNETES,
            start_time=datetime.now()
        )
        
        if not self.apps_v1 or not self.core_v1:
            result.status = DeploymentStatus.FAILED
            result.errors.append("Kubernetes client not available")
            return result
        
        try:
            namespace = self.config.environment.get("KUBERNETES_NAMESPACE", "default")
            
            # Create deployment manifests
            result.status = DeploymentStatus.BUILDING
            deployment_manifest = self._create_deployment_manifest(namespace, result.deployment_id)
            service_manifest = self._create_service_manifest(namespace)
            
            # Apply manifests
            result.status = DeploymentStatus.DEPLOYING
            self.logger.info(f"Deploying to Kubernetes namespace: {namespace}")
            
            # Create or update deployment
            try:
                self.apps_v1.read_namespaced_deployment(name=self.config.name, namespace=namespace)
                # Deployment exists, update it
                self.apps_v1.patch_namespaced_deployment(
                    name=self.config.name,
                    namespace=namespace,
                    body=deployment_manifest
                )
                self.logger.info(f"Updated existing deployment: {self.config.name}")
            except client.rest.ApiException as e:
                if e.status == 404:
                    # Deployment doesn't exist, create it
                    self.apps_v1.create_namespaced_deployment(
                        namespace=namespace,
                        body=deployment_manifest
                    )
                    self.logger.info(f"Created new deployment: {self.config.name}")
                else:
                    raise e
            
            # Create or update service
            try:
                self.core_v1.read_namespaced_service(name=self.config.name, namespace=namespace)
                # Service exists, update it
                self.core_v1.patch_namespaced_service(
                    name=self.config.name,
                    namespace=namespace,
                    body=service_manifest
                )
            except client.rest.ApiException as e:
                if e.status == 404:
                    # Service doesn't exist, create it
                    self.core_v1.create_namespaced_service(
                        namespace=namespace,
                        body=service_manifest
                    )
                else:
                    raise e
            
            # Wait for deployment to be ready
            await self._wait_for_deployment_ready(namespace, result)
            
            if result.status != DeploymentStatus.FAILED:
                result.status = DeploymentStatus.RUNNING
                
                # Get service URLs
                service_urls = await self._get_service_urls(namespace)
                result.urls = service_urls
                
                # Setup monitoring if enabled
                if self.config.monitoring.enable_metrics:
                    await self._setup_k8s_monitoring(namespace, result)
                
                # Setup auto-scaling if configured
                if self.config.scaling:
                    await self._setup_horizontal_pod_autoscaler(namespace, result)
            
            result.end_time = datetime.now()
            self.logger.info(f"Kubernetes deployment completed: {result.deployment_id}")
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.errors.append(f"Kubernetes deployment failed: {str(e)}")
            result.end_time = datetime.now()
            self.logger.error(f"Kubernetes deployment error: {e}")
        
        return result
    
    def _create_deployment_manifest(self, namespace: str, deployment_id: str) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest"""
        
        containers = [{
            "name": self.config.name,
            "image": self.config.image or f"{self.config.name}:latest",
            "ports": [{"containerPort": self.config.port}],
            "env": [{"name": k, "value": str(v)} for k, v in self.config.environment.items()],
            "resources": {
                "requests": {
                    "cpu": self.config.resources.get("cpu", "100m"),
                    "memory": self.config.resources.get("memory", "128Mi")
                },
                "limits": {
                    "cpu": self.config.resources.get("cpu", "500m"),
                    "memory": self.config.resources.get("memory", "512Mi")
                }
            }
        }]
        
        # Add GPU resources if specified
        if self.config.resources.get("gpu"):
            containers[0]["resources"]["limits"]["nvidia.com/gpu"] = str(self.config.resources["gpu"])
        
        # Add health checks
        if self.config.health_checks:
            for health_check in self.config.health_checks:
                if health_check.type == HealthCheckType.HTTP:
                    containers[0]["livenessProbe"] = {
                        "httpGet": {
                            "path": health_check.endpoint or "/health",
                            "port": self.config.port
                        },
                        "initialDelaySeconds": health_check.initial_delay,
                        "periodSeconds": health_check.interval,
                        "timeoutSeconds": health_check.timeout,
                        "failureThreshold": health_check.retries
                    }
                    containers[0]["readinessProbe"] = {
                        "httpGet": {
                            "path": health_check.endpoint or "/ready",
                            "port": self.config.port
                        },
                        "initialDelaySeconds": 5,
                        "periodSeconds": 10
                    }
        
        # Add volume mounts
        if self.config.volumes:
            containers[0]["volumeMounts"] = [
                {
                    "name": f"volume-{i}",
                    "mountPath": vol["container_path"],
                    "readOnly": vol.get("mode", "rw") == "ro"
                }
                for i, vol in enumerate(self.config.volumes)
            ]
        
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.config.name,
                "namespace": namespace,
                "labels": {
                    "app": self.config.name,
                    "nis.deployment.id": deployment_id,
                    "nis.deployment.platform": "kubernetes"
                },
                "annotations": self.config.custom_annotations
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {"app": self.config.name}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": self.config.name}
                    },
                    "spec": {
                        "containers": containers
                    }
                }
            }
        }
        
        # Add volumes
        if self.config.volumes:
            manifest["spec"]["template"]["spec"]["volumes"] = [
                {
                    "name": f"volume-{i}",
                    "hostPath": {"path": vol["host_path"]}
                }
                for i, vol in enumerate(self.config.volumes)
            ]
        
        return manifest
    
    def _create_service_manifest(self, namespace: str) -> Dict[str, Any]:
        """Create Kubernetes service manifest"""
        
        service_type = "ClusterIP"
        if self.config.environment.get("KUBERNETES_SERVICE_TYPE"):
            service_type = self.config.environment["KUBERNETES_SERVICE_TYPE"]
        
        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": self.config.name,
                "namespace": namespace,
                "labels": {"app": self.config.name}
            },
            "spec": {
                "selector": {"app": self.config.name},
                "ports": [{
                    "port": 80,
                    "targetPort": self.config.port,
                    "protocol": "TCP"
                }],
                "type": service_type
            }
        }
        
        return manifest
    
    async def _wait_for_deployment_ready(self, namespace: str, result: DeploymentResult):
        """Wait for Kubernetes deployment to be ready"""
        
        max_wait = 600  # 10 minutes
        wait_interval = 10
        elapsed = 0
        
        while elapsed < max_wait:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(
                    name=self.config.name,
                    namespace=namespace
                )
                
                if deployment.status.ready_replicas == deployment.spec.replicas:
                    self.logger.info(f"Deployment {self.config.name} is ready")
                    return
                
                self.logger.info(f"Waiting for deployment {self.config.name}: "
                              f"{deployment.status.ready_replicas or 0}/{deployment.spec.replicas} ready")
                
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval
                
            except Exception as e:
                result.errors.append(f"Deployment readiness check error: {str(e)}")
                break
        
        # Timeout reached
        result.status = DeploymentStatus.FAILED
        result.errors.append("Deployment readiness timeout")
    
    async def _get_service_urls(self, namespace: str) -> List[str]:
        """Get service URLs"""
        
        urls = []
        try:
            service = self.core_v1.read_namespaced_service(name=self.config.name, namespace=namespace)
            
            if service.spec.type == "LoadBalancer":
                if service.status.load_balancer.ingress:
                    for ingress in service.status.load_balancer.ingress:
                        if ingress.ip:
                            urls.append(f"http://{ingress.ip}")
                        elif ingress.hostname:
                            urls.append(f"http://{ingress.hostname}")
            
            elif service.spec.type == "NodePort":
                # Get node IPs
                nodes = self.core_v1.list_node()
                for node in nodes.items:
                    for address in node.status.addresses:
                        if address.type == "ExternalIP":
                            for port in service.spec.ports:
                                if port.node_port:
                                    urls.append(f"http://{address.address}:{port.node_port}")
                            break
            
            else:  # ClusterIP
                urls.append(f"http://{service.spec.cluster_ip}")
        
        except Exception as e:
            self.logger.warning(f"Failed to get service URLs: {e}")
        
        return urls
    
    async def _setup_k8s_monitoring(self, namespace: str, result: DeploymentResult):
        """Setup monitoring for Kubernetes deployment"""
        
        try:
            # Get pod metrics
            pods = self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"app={self.config.name}"
            )
            
            result.metrics.update({
                "namespace": namespace,
                "pod_count": len(pods.items),
                "pod_names": [pod.metadata.name for pod in pods.items],
                "pod_statuses": {pod.metadata.name: pod.status.phase for pod in pods.items}
            })
            
        except Exception as e:
            self.logger.warning(f"Failed to setup Kubernetes monitoring: {e}")
    
    async def _setup_horizontal_pod_autoscaler(self, namespace: str, result: DeploymentResult):
        """Setup Horizontal Pod Autoscaler"""
        
        if not self.config.scaling:
            return
        
        try:
            hpa_manifest = {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": f"{self.config.name}-hpa",
                    "namespace": namespace
                },
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": self.config.name
                    },
                    "minReplicas": self.config.scaling.min_replicas,
                    "maxReplicas": self.config.scaling.max_replicas,
                    "metrics": [
                        {
                            "type": "Resource",
                            "resource": {
                                "name": "cpu",
                                "target": {
                                    "type": "Utilization",
                                    "averageUtilization": self.config.scaling.target_cpu_utilization
                                }
                            }
                        },
                        {
                            "type": "Resource",
                            "resource": {
                                "name": "memory",
                                "target": {
                                    "type": "Utilization",
                                    "averageUtilization": self.config.scaling.target_memory_utilization
                                }
                            }
                        }
                    ]
                }
            }
            
            # Create or update HPA
            autoscaling_v2 = client.AutoscalingV2Api()
            try:
                autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                    name=f"{self.config.name}-hpa",
                    namespace=namespace
                )
                # HPA exists, update it
                autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(
                    name=f"{self.config.name}-hpa",
                    namespace=namespace,
                    body=hpa_manifest
                )
            except client.rest.ApiException as e:
                if e.status == 404:
                    # HPA doesn't exist, create it
                    autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                        namespace=namespace,
                        body=hpa_manifest
                    )
            
            self.logger.info(f"Horizontal Pod Autoscaler configured for {self.config.name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup HPA: {e}")

class NISDeploymentOrchestrator:
    """
    Main deployment orchestrator for NIS-powered systems
    
    Features:
    - Multi-platform deployment support
    - Intelligent platform selection
    - Rollback capabilities
    - Health monitoring
    - Auto-scaling
    - Security configuration
    - Monitoring integration
    """
    
    def __init__(self):
        self.deployment_history = []
        self.active_deployments = {}
        self.platform_deployers = {
            DeploymentPlatform.DOCKER: DockerDeployer,
            DeploymentPlatform.KUBERNETES: KubernetesDeployer,
            # Additional deployers can be added here
        }
        self.logger = logging.getLogger(f"{__name__}.NISDeploymentOrchestrator")
    
    async def deploy(self, config: DeploymentConfig, dry_run: bool = False) -> DeploymentResult:
        """Deploy NIS system with comprehensive configuration"""
        
        self.logger.info(f"Starting deployment: {config.name} to {config.platform.value}")
        
        if dry_run:
            return await self._simulate_deployment(config)
        
        # Validate configuration
        validation_result = await self._validate_deployment_config(config)
        if not validation_result["valid"]:
            result = DeploymentResult(
                deployment_id=f"failed_{config.name}_{int(datetime.now().timestamp())}",
                status=DeploymentStatus.FAILED,
                platform=config.platform,
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            result.errors.extend(validation_result["errors"])
            return result
        
        # Generate default configurations if needed
        await self._enhance_deployment_config(config)
        
        # Get platform deployer
        deployer_class = self.platform_deployers.get(config.platform)
        if not deployer_class:
            result = DeploymentResult(
                deployment_id=f"unsupported_{config.name}_{int(datetime.now().timestamp())}",
                status=DeploymentStatus.FAILED,
                platform=config.platform,
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            result.errors.append(f"Unsupported platform: {config.platform.value}")
            return result
        
        # Create deployer and execute deployment
        deployer = deployer_class(config)
        result = await deployer.deploy()
        
        # Store deployment result
        self.deployment_history.append(result)
        if result.status == DeploymentStatus.RUNNING:
            self.active_deployments[config.name] = result
        
        # Setup post-deployment monitoring
        if result.status == DeploymentStatus.RUNNING and config.monitoring.enable_metrics:
            await self._setup_post_deployment_monitoring(config, result)
        
        self.logger.info(f"Deployment completed: {result.deployment_id} - {result.status.value}")
        return result
    
    async def _simulate_deployment(self, config: DeploymentConfig) -> DeploymentResult:
        """Simulate deployment for dry run"""
        
        result = DeploymentResult(
            deployment_id=f"simulation_{config.name}_{int(datetime.now().timestamp())}",
            status=DeploymentStatus.RUNNING,
            platform=config.platform,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Simulate deployment process
        result.logs.extend([
            f"[SIMULATION] Building image for {config.name}",
            f"[SIMULATION] Deploying to {config.platform.value}",
            f"[SIMULATION] Configuring {config.replicas} replicas",
            f"[SIMULATION] Setting up health checks",
            f"[SIMULATION] Enabling monitoring",
            f"[SIMULATION] Deployment would succeed"
        ])
        
        result.urls = [f"http://localhost:{config.port}"]
        result.metrics = {
            "simulated": True,
            "estimated_deployment_time": "2-5 minutes",
            "estimated_resources": config.resources
        }
        
        return result
    
    async def _validate_deployment_config(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Validate deployment configuration"""
        
        errors = []
        warnings = []
        
        # Basic validation
        if not config.name:
            errors.append("Deployment name is required")
        
        if not config.name.replace("-", "").replace("_", "").isalnum():
            errors.append("Deployment name must be alphanumeric (with hyphens/underscores)")
        
        if config.port < 1 or config.port > 65535:
            errors.append("Port must be between 1 and 65535")
        
        if config.replicas < 1:
            errors.append("Replicas must be at least 1")
        
        # Platform-specific validation
        if config.platform == DeploymentPlatform.DOCKER and not DOCKER_AVAILABLE:
            errors.append("Docker client not available for Docker deployment")
        
        if config.platform == DeploymentPlatform.KUBERNETES and not KUBERNETES_AVAILABLE:
            errors.append("Kubernetes client not available for Kubernetes deployment")
        
        # Resource validation
        if config.resources:
            cpu = config.resources.get("cpu", "")
            if cpu and not any(cpu.endswith(unit) for unit in ["m", ""]):
                warnings.append("CPU resource format should end with 'm' for millicores")
            
            memory = config.resources.get("memory", "")
            if memory and not any(memory.endswith(unit) for unit in ["Mi", "Gi", "Ki"]):
                warnings.append("Memory resource format should end with 'Mi', 'Gi', or 'Ki'")
        
        # Security validation
        if config.security.enable_tls and not config.security.certificate_mode:
            warnings.append("TLS enabled but no certificate mode specified")
        
        # Scaling validation
        if config.scaling:
            if config.scaling.min_replicas > config.scaling.max_replicas:
                errors.append("Minimum replicas cannot be greater than maximum replicas")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _enhance_deployment_config(self, config: DeploymentConfig):
        """Enhance deployment configuration with intelligent defaults"""
        
        # Add default environment variables
        if "NIS_ENVIRONMENT" not in config.environment:
            config.environment["NIS_ENVIRONMENT"] = "production"
        
        if "NIS_PORT" not in config.environment:
            config.environment["NIS_PORT"] = str(config.port)
        
        # Add consciousness monitoring environment variables
        if config.monitoring.consciousness_monitoring:
            config.environment.update({
                "NIS_CONSCIOUSNESS_MONITORING": "true",
                "NIS_CONSCIOUSNESS_ENDPOINT": "/consciousness",
                "NIS_PERFORMANCE_MONITORING": "true"
            })
        
        # Add default health checks if none specified
        if not config.health_checks:
            config.health_checks = [
                HealthCheck(
                    type=HealthCheckType.HTTP,
                    endpoint="/health",
                    interval=30,
                    timeout=5,
                    retries=3
                )
            ]
            
            # Add consciousness health check if monitoring is enabled
            if config.monitoring.consciousness_monitoring:
                config.health_checks.append(
                    HealthCheck(
                        type=HealthCheckType.CONSCIOUSNESS,
                        endpoint="/consciousness/health",
                        interval=60,
                        timeout=10,
                        retries=2
                    )
                )
        
        # Add default scaling if not specified but replicas > 1
        if not config.scaling and config.replicas > 1:
            config.scaling = ScalingConfig(
                min_replicas=1,
                max_replicas=config.replicas * 2,
                target_cpu_utilization=70,
                target_consciousness_load=0.8
            )
        
        # Add default security labels
        config.custom_annotations.update({
            "nis.ai/framework": "nis-protocol",
            "nis.ai/deployment-time": datetime.now().isoformat(),
            "nis.ai/consciousness-monitoring": str(config.monitoring.consciousness_monitoring).lower()
        })
    
    async def _setup_post_deployment_monitoring(self, config: DeploymentConfig, result: DeploymentResult):
        """Setup post-deployment monitoring and observability"""
        
        try:
            # Create monitoring configuration
            monitoring_config = {
                "deployment_id": result.deployment_id,
                "platform": config.platform.value,
                "endpoints": result.urls,
                "health_checks": [
                    {
                        "type": hc.type.value,
                        "endpoint": hc.endpoint,
                        "interval": hc.interval
                    }
                    for hc in config.health_checks
                ],
                "consciousness_monitoring": config.monitoring.consciousness_monitoring,
                "performance_monitoring": config.monitoring.performance_monitoring,
                "custom_metrics": config.monitoring.custom_metrics
            }
            
            # Store monitoring configuration for external monitoring systems
            result.metrics["monitoring_config"] = monitoring_config
            
            self.logger.info(f"Monitoring configured for deployment: {result.deployment_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup post-deployment monitoring: {e}")
    
    async def rollback(self, deployment_name: str) -> DeploymentResult:
        """Rollback deployment to previous version"""
        
        self.logger.info(f"Starting rollback for deployment: {deployment_name}")
        
        # Find current deployment
        current_deployment = self.active_deployments.get(deployment_name)
        if not current_deployment:
            result = DeploymentResult(
                deployment_id=f"rollback_failed_{deployment_name}_{int(datetime.now().timestamp())}",
                status=DeploymentStatus.FAILED,
                platform=DeploymentPlatform.LOCAL,  # Default
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            result.errors.append(f"No active deployment found: {deployment_name}")
            return result
        
        # Find previous successful deployment
        previous_deployments = [
            d for d in self.deployment_history
            if d.platform == current_deployment.platform and 
               deployment_name in d.deployment_id and
               d.status == DeploymentStatus.RUNNING and
               d.deployment_id != current_deployment.deployment_id
        ]
        
        if not previous_deployments:
            result = DeploymentResult(
                deployment_id=f"rollback_failed_{deployment_name}_{int(datetime.now().timestamp())}",
                status=DeploymentStatus.FAILED,
                platform=current_deployment.platform,
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            result.errors.append(f"No previous deployment found for rollback: {deployment_name}")
            return result
        
        # Get the most recent previous deployment
        previous_deployment = sorted(previous_deployments, key=lambda x: x.start_time)[-1]
        
        # Create rollback deployment result
        result = DeploymentResult(
            deployment_id=f"rollback_{deployment_name}_{int(datetime.now().timestamp())}",
            status=DeploymentStatus.ROLLING_BACK,
            platform=current_deployment.platform,
            start_time=datetime.now()
        )
        
        result.rollback_info = {
            "from_deployment": current_deployment.deployment_id,
            "to_deployment": previous_deployment.deployment_id,
            "rollback_reason": "manual_rollback"
        }
        
        try:
            # Platform-specific rollback logic would go here
            # For now, simulate rollback
            await asyncio.sleep(2)  # Simulate rollback time
            
            result.status = DeploymentStatus.RUNNING
            result.end_time = datetime.now()
            result.urls = previous_deployment.urls
            
            # Update active deployments
            self.active_deployments[deployment_name] = result
            
            self.logger.info(f"Rollback completed: {result.deployment_id}")
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.errors.append(f"Rollback failed: {str(e)}")
            result.end_time = datetime.now()
            self.logger.error(f"Rollback error: {e}")
        
        self.deployment_history.append(result)
        return result
    
    def get_deployment_status(self, deployment_name: str) -> Optional[DeploymentResult]:
        """Get status of a deployment"""
        return self.active_deployments.get(deployment_name)
    
    def list_deployments(self) -> List[DeploymentResult]:
        """List all active deployments"""
        return list(self.active_deployments.values())
    
    def get_deployment_history(self, deployment_name: str = None) -> List[DeploymentResult]:
        """Get deployment history"""
        if deployment_name:
            return [d for d in self.deployment_history if deployment_name in d.deployment_id]
        return self.deployment_history

# Factory functions

def create_deployment_config(name: str, platform: str, **kwargs) -> DeploymentConfig:
    """Create deployment configuration with intelligent defaults"""
    
    platform_enum = DeploymentPlatform(platform.lower())
    
    config = DeploymentConfig(
        name=name,
        platform=platform_enum,
        **kwargs
    )
    
    return config

def create_deployment_orchestrator() -> NISDeploymentOrchestrator:
    """Create NIS deployment orchestrator"""
    return NISDeploymentOrchestrator()

# Example usage
async def example_deployment():
    """Example deployment workflow"""
    
    # Create deployment configuration
    config = create_deployment_config(
        name="nis-reasoning-service",
        platform="docker",
        port=8000,
        replicas=2,
        environment={
            "NIS_MODE": "production",
            "CONSCIOUSNESS_LEVEL": "0.8"
        },
        resources={
            "cpu": "500m",
            "memory": "512Mi"
        },
        monitoring=MonitoringConfig(
            consciousness_monitoring=True,
            performance_monitoring=True,
            enable_metrics=True
        ),
        scaling=ScalingConfig(
            min_replicas=1,
            max_replicas=5,
            target_cpu_utilization=70,
            target_consciousness_load=0.8
        )
    )
    
    # Create orchestrator and deploy
    orchestrator = create_deployment_orchestrator()
    
    # Dry run first
    dry_run_result = await orchestrator.deploy(config, dry_run=True)
    print(f"Dry run result: {dry_run_result.status}")
    
    # Actual deployment
    result = await orchestrator.deploy(config)
    print(f"Deployment result: {result.status}")
    print(f"URLs: {result.urls}")
    print(f"Metrics: {result.metrics}")
    
    # Check status
    status = orchestrator.get_deployment_status(config.name)
    print(f"Current status: {status.status if status else 'Not found'}")

if __name__ == "__main__":
    asyncio.run(example_deployment())
