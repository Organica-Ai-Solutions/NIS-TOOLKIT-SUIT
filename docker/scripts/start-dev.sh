#!/bin/bash

# NIS TOOLKIT SUIT v3.2.1 - Development Environment Startup
# Quick start script for development with hot-reload

set -e

echo "🚀 Starting NIS TOOLKIT SUIT v3.2.1 Development Environment"
echo "================================================================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose >/dev/null 2>&1; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs data cache

# Build and start development environment
echo "🏗️  Building development containers..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml build

echo "🚀 Starting development services..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Health check
echo "🏥 Checking service health..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps

# Show access URLs
echo ""
echo "✅ NIS TOOLKIT SUIT Development Environment is ready!"
echo "================================================================"
echo "🎯 Core Service:      http://localhost:8000"
echo "🤖 Agent Service:     http://localhost:8001"
echo "📱 Edge Service:      http://localhost:8002"
echo "📊 Prometheus:        http://localhost:9090"
echo "📈 Grafana:           http://localhost:3000 (admin/nis_admin_2024)"
echo "📋 Jupyter Lab:       http://localhost:8888 (token: nis_toolkit_2024)"
echo "📁 Dev Files:         http://localhost:8090"
echo ""
echo "📋 Useful Commands:"
echo "  View logs:          docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f"
echo "  Stop services:      docker-compose -f docker-compose.yml -f docker-compose.dev.yml down"
echo "  Restart core:       docker-compose -f docker-compose.yml -f docker-compose.dev.yml restart nis-core"
echo ""
echo "🎉 Happy developing!"
