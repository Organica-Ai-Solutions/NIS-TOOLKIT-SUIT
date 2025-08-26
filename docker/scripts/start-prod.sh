#!/bin/bash

# NIS TOOLKIT SUIT v3.2.1 - Production Environment Startup
# Production deployment with monitoring and scaling

set -e

echo "🏭 Starting NIS TOOLKIT SUIT v3.2.1 Production Environment"
echo "============================================================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Production environment checks
echo "🔍 Running production readiness checks..."

# Check available resources
MEMORY=$(docker system info --format '{{.MemTotal}}' 2>/dev/null || echo "0")
if [ "$MEMORY" -lt 4294967296 ]; then  # 4GB in bytes
    echo "⚠️  Warning: Less than 4GB RAM available. Production deployment may be constrained."
fi

# Check disk space
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "⚠️  Warning: Disk usage is ${DISK_USAGE}%. Consider freeing up space."
fi

# Create production directories with proper permissions
echo "📁 Setting up production directories..."
mkdir -p logs data cache docker/nginx/ssl
chmod 755 logs data cache

# Generate SSL certificates if they don't exist
if [ ! -f "docker/nginx/ssl/nginx.crt" ]; then
    echo "🔒 Generating self-signed SSL certificates..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout docker/nginx/ssl/nginx.key \
        -out docker/nginx/ssl/nginx.crt \
        -subj "/C=US/ST=State/L=City/O=NIS/CN=localhost" 2>/dev/null || \
        echo "⚠️  OpenSSL not available. SSL certificates not generated."
fi

# Build production images
echo "🏗️  Building production containers..."
docker-compose build --no-cache

# Start production services
echo "🚀 Starting production services..."
docker-compose --profile production up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 15

# Health checks
echo "🏥 Running health checks..."
MAX_RETRIES=10
RETRY_COUNT=0

check_service() {
    local service_name=$1
    local port=$2
    local endpoint=${3:-/health}
    
    echo "   Checking $service_name..."
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -sf "http://localhost:$port$endpoint" >/dev/null 2>&1; then
            echo "   ✅ $service_name is healthy"
            return 0
        fi
        sleep 2
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done
    echo "   ❌ $service_name health check failed"
    return 1
}

# Check core services
check_service "NIS Core" 8000
check_service "Prometheus" 9090 "/api/v1/status/config"
check_service "Grafana" 3000 "/api/health"

# Show deployment status
echo ""
echo "📊 Deployment Status:"
docker-compose ps

# Show access URLs
echo ""
echo "✅ NIS TOOLKIT SUIT Production Environment is ready!"
echo "======================================================="
echo "🎯 Core Service:      https://localhost (or http://localhost:8000)"
echo "🤖 Agent Service:     http://localhost:8001"
echo "📱 Edge Service:      http://localhost:8002"
echo "📊 Prometheus:        http://localhost:9090"
echo "📈 Grafana:           http://localhost:3000 (admin/nis_admin_2024)"
echo "🔒 SSL Enabled:       https://localhost"
echo ""
echo "📋 Production Commands:"
echo "  View logs:          docker-compose logs -f"
echo "  Scale core:         docker-compose up -d --scale nis-core=3"
echo "  Stop services:      docker-compose down"
echo "  Update images:      docker-compose pull && docker-compose up -d"
echo ""
echo "🏭 Production deployment complete!"
