#!/bin/bash

# NIS TOOLKIT SUIT v3.2.1 - Docker Cleanup Script
# Clean up containers, images, volumes, and networks

set -e

echo "🧹 NIS TOOLKIT SUIT Docker Cleanup"
echo "==================================="

# Function to confirm action
confirm() {
    read -p "$1 (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# Stop all running containers
if confirm "Stop all NIS TOOLKIT containers?"; then
    echo "🛑 Stopping containers..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml down 2>/dev/null || true
    docker-compose down 2>/dev/null || true
    
    # Force stop any remaining containers
    NIS_CONTAINERS=$(docker ps -q --filter "name=nis-*")
    if [ ! -z "$NIS_CONTAINERS" ]; then
        docker stop $NIS_CONTAINERS
    fi
fi

# Remove containers
if confirm "Remove NIS TOOLKIT containers?"; then
    echo "🗑️  Removing containers..."
    NIS_CONTAINERS=$(docker ps -aq --filter "name=nis-*")
    if [ ! -z "$NIS_CONTAINERS" ]; then
        docker rm $NIS_CONTAINERS
    fi
fi

# Remove images
if confirm "Remove NIS TOOLKIT images?"; then
    echo "🗑️  Removing images..."
    NIS_IMAGES=$(docker images --filter "reference=*nis*" -q)
    if [ ! -z "$NIS_IMAGES" ]; then
        docker rmi $NIS_IMAGES
    fi
    
    # Remove dangling images
    DANGLING_IMAGES=$(docker images -f "dangling=true" -q)
    if [ ! -z "$DANGLING_IMAGES" ]; then
        docker rmi $DANGLING_IMAGES
    fi
fi

# Remove volumes
if confirm "Remove NIS TOOLKIT volumes? (⚠️  This will delete all data)"; then
    echo "🗑️  Removing volumes..."
    NIS_VOLUMES=$(docker volume ls -q --filter "name=nis-*")
    if [ ! -z "$NIS_VOLUMES" ]; then
        docker volume rm $NIS_VOLUMES
    fi
fi

# Remove networks
if confirm "Remove NIS TOOLKIT networks?"; then
    echo "🗑️  Removing networks..."
    NIS_NETWORKS=$(docker network ls -q --filter "name=nis-*")
    if [ ! -z "$NIS_NETWORKS" ]; then
        docker network rm $NIS_NETWORKS
    fi
fi

# System cleanup
if confirm "Run Docker system cleanup?"; then
    echo "🧹 Running system cleanup..."
    docker system prune -f
    docker volume prune -f
    docker network prune -f
fi

# Show remaining Docker resources
echo ""
echo "📊 Remaining Docker Resources:"
echo "Containers: $(docker ps -a | wc -l)"
echo "Images: $(docker images | wc -l)"
echo "Volumes: $(docker volume ls | wc -l)"
echo "Networks: $(docker network ls | wc -l)"

# Show disk space freed
echo ""
echo "💾 Docker Disk Usage:"
docker system df

echo ""
echo "✅ Cleanup complete!"
