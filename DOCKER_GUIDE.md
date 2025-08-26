# 🐳 NIS TOOLKIT SUIT v3.2.1 - Docker Deployment Guide

## 🚀 **Quick Start**

### **Development Environment (Hot-reload)**
```bash
# Start development environment with hot-reload
./docker/scripts/start-dev.sh

# Or manually:
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### **Production Environment**
```bash
# Start production environment with monitoring
./docker/scripts/start-prod.sh

# Or manually:
docker-compose --profile production up -d
```

### **Edge Deployment**
```bash
# Build and run edge-optimized container
docker build --target edge -t nis-edge:latest .
docker run -p 8002:8000 nis-edge:latest
```

---

## 📋 **Container Architecture**

### **Multi-Stage Dockerfile**
- **Base Stage**: Common dependencies and security setup
- **Development Stage**: Hot-reload, debugging tools, Jupyter
- **Production Stage**: Optimized, minimal attack surface
- **Edge Stage**: Ultra-lightweight for IoT/embedded

### **Service Stack**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Monitoring    │    │   Development   │
│    (Nginx)      │    │ (Prometheus +   │    │   (Jupyter)     │
│    Port 80/443  │    │  Grafana)       │    │   Port 8888     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────┬─────────────────┬─────────────────────────────┐
│                 │                 │                             │
├─────────────────┼─────────────────┼─────────────────┐           │
│   NIS Core      │   NIS Agent     │   NIS Edge      │           │
│   Port 8000     │   Port 8001     │   Port 8002     │           │
└─────────────────┴─────────────────┴─────────────────┘           │
         │                 │                 │                   │
┌─────────────────┬─────────────────┬─────────────────────────────┘
│   Redis Cache   │   Kafka Queue   │   Infrastructure
│   Port 6379     │   Port 9092     │   (Zookeeper, etc.)
└─────────────────┴─────────────────┘
```

---

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Core Configuration
NIS_ENVIRONMENT=docker|development|production|edge
NIS_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
NIS_ROLE=core|agent|edge

# Infrastructure
REDIS_HOST=redis
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ENABLE_MONITORING=true|false

# Development
DEVELOPMENT_MODE=true|false
HOT_RELOAD=true|false
PYTHONPATH=/app
```

### **Volumes**
- **nis-data**: Persistent application data
- **nis-logs**: Application logs
- **nis-cache**: Temporary cache data
- **redis-data**: Redis persistence
- **kafka-data**: Kafka logs and data

### **Ports**
- **8000**: NIS Core Service
- **8001**: NIS Agent Service
- **8002**: NIS Edge Service
- **6379**: Redis Cache
- **9092**: Kafka Message Queue
- **9090**: Prometheus Metrics
- **3000**: Grafana Dashboards
- **8888**: Jupyter Lab (dev only)
- **80/443**: Nginx Load Balancer (production)

---

## 🏗️ **Build Options**

### **Development Build**
```bash
# With hot-reload and debugging
docker build --target development -t nis-dev:latest .
```

### **Production Build**
```bash
# Optimized for production
docker build --target production -t nis-prod:latest .
```

### **Edge Build**
```bash
# Minimal for edge/IoT devices
docker build --target edge -t nis-edge:latest .
```

### **Custom Build**
```bash
# With build arguments
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg ENVIRONMENT=production \
  -t nis-custom:latest .
```

---

## 📊 **Monitoring & Observability**

### **Access Dashboards**
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/nis_admin_2024)
- **Node Exporter**: http://localhost:9100/metrics

### **Key Metrics**
- **Container Health**: CPU, Memory, Network usage
- **Application Metrics**: Request rates, response times, error rates
- **Infrastructure**: Redis hits/misses, Kafka throughput
- **Custom Metrics**: AI model performance, consciousness scores

### **Alerting**
Configure alerts in Prometheus for:
- High memory usage (>80%)
- High CPU usage (>90%)
- Service downtime
- Error rate spikes
- Disk space low (<10%)

---

## 🔒 **Security Features**

### **Container Security**
- Non-root user execution
- Minimal base images
- Security-hardened dependencies
- Health checks and resource limits

### **Network Security**
- Internal Docker networks
- SSL termination at load balancer
- Rate limiting on API endpoints
- Security headers

### **Secrets Management**
```bash
# Use Docker secrets for sensitive data
echo "your-api-key" | docker secret create nis_api_key -
```

---

## 🚀 **Kubernetes Deployment**

### **Apply Kubernetes Configuration**
```bash
# Deploy to Kubernetes
kubectl apply -f docker/kubernetes/nis-deployment.yaml

# Check deployment status
kubectl get pods -n nis-toolkit
kubectl get services -n nis-toolkit
```

### **Scaling**
```bash
# Manual scaling
kubectl scale deployment nis-core --replicas=5 -n nis-toolkit

# Auto-scaling (HPA already configured)
kubectl get hpa -n nis-toolkit
```

### **Access Services**
```bash
# Port forward for local access
kubectl port-forward service/nis-core-service 8000:80 -n nis-toolkit
```

---

## 🧹 **Maintenance**

### **Update Containers**
```bash
# Pull latest images and restart
docker-compose pull
docker-compose up -d

# For Kubernetes
kubectl rollout restart deployment/nis-core -n nis-toolkit
```

### **Backup Data**
```bash
# Backup volumes
docker run --rm -v nis-data:/data -v $(pwd):/backup alpine tar czf /backup/nis-data-backup.tar.gz /data
```

### **Cleanup**
```bash
# Complete cleanup
./docker/scripts/cleanup.sh

# Or selective cleanup
docker system prune -f
```

### **Logs Management**
```bash
# View logs
docker-compose logs -f nis-core

# Log rotation (configure in docker-compose.yml)
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Port Conflicts**
   ```bash
   # Check what's using the port
   lsof -i :8000
   
   # Use different ports
   docker-compose up -d --scale nis-core=0
   ```

2. **Memory Issues**
   ```bash
   # Check Docker memory
   docker system df
   docker stats
   
   # Increase memory limits in docker-compose.yml
   ```

3. **Permission Issues**
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 data logs cache
   ```

4. **Network Issues**
   ```bash
   # Check network connectivity
   docker network ls
   docker exec nis-core curl http://redis:6379
   ```

### **Debug Commands**
```bash
# Enter container for debugging
docker exec -it nis-core bash

# Check container health
docker inspect nis-core | grep Health

# View container processes
docker exec nis-core ps aux
```

---

## 🎯 **Performance Tuning**

### **Resource Optimization**
- Adjust memory/CPU limits based on usage
- Use multi-stage builds to minimize image size
- Enable compression in Nginx
- Configure Redis memory policies

### **Scaling Strategies**
- Horizontal scaling with load balancer
- Auto-scaling based on metrics
- Edge deployment for regional distribution
- Database read replicas

---

## 📚 **Additional Resources**

- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)

---

✅ **Your NIS TOOLKIT SUIT is now containerized and production-ready!**
