#!/usr/bin/env python3
"""
NIS-TOOLKIT-SUIT Ecosystem Integration Example
Demonstrates how to integrate with the complete Organica AI ecosystem
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

# Import NIS Toolkit components
from nis_core_toolkit.integration import create_nis_integration_connector, MessageType
from nis_agent_toolkit.core.base_agent import NISAgent, AgentCapability
from nis_core_toolkit.core.tool_loader import ToolLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcosystemIntegrationDemo:
    """
    Complete demonstration of NIS ecosystem integration
    Shows how to connect to multiple NIS projects and coordinate activities
    """
    
    def __init__(self):
        self.connector = None
        self.agents = {}
        self.connected_projects = []
        self.integration_history = []
    
    async def initialize_ecosystem_connection(self):
        """Initialize connection to the NIS ecosystem"""
        
        logger.info("🚀 Initializing NIS Ecosystem Connection...")
        
        # Create integration connector
        self.connector = create_nis_integration_connector(
            connector_id="ECOSYSTEM-DEMO-CONNECTOR",
            config={
                "max_connections": 20,
                "authentication_enabled": True,
                "heartbeat_interval": 30,
                "message_timeout": 60
            }
        )
        
        logger.info("✅ Integration connector created")
    
    async def connect_to_all_projects(self):
        """Connect to all available NIS projects in the ecosystem"""
        
        logger.info("🔗 Connecting to NIS ecosystem projects...")
        
        # Define ecosystem projects with their connection details
        ecosystem_projects = {
            "nis-hub": {
                "endpoint_url": "https://api.nis-hub.org/v1",
                "websocket_url": "wss://ws.nis-hub.org/v1",
                "api_key": "demo-nis-hub-key",
                "api_secret": "demo-nis-hub-secret",
                "capabilities": ["system_orchestration", "workflow_management", "agent_coordination"]
            },
            "nis-x": {
                "endpoint_url": "https://api.nis-x.space/v1",
                "websocket_url": "wss://ws.nis-x.space/v1", 
                "api_key": "demo-nis-x-key",
                "api_secret": "demo-nis-x-secret",
                "capabilities": ["orbital_navigation", "mission_planning", "spacecraft_control"]
            },
            "nis-drone": {
                "endpoint_url": "https://api.nis-drone.com/v1",
                "websocket_url": "wss://ws.nis-drone.com/v1",
                "api_key": "demo-nis-drone-key", 
                "api_secret": "demo-nis-drone-secret",
                "capabilities": ["swarm_coordination", "formation_flight", "autonomous_navigation"]
            },
            "sparknova": {
                "endpoint_url": "https://api.sparknova.dev/v1",
                "websocket_url": "wss://ws.sparknova.dev/v1",
                "api_key": "demo-sparknova-key",
                "api_secret": "demo-sparknova-secret",
                "capabilities": ["development_tools", "agent_marketplace", "code_generation"]
            },
            "orion": {
                "endpoint_url": "https://api.orion-llm.com/v1",
                "websocket_url": "wss://ws.orion-llm.com/v1",
                "api_key": "demo-orion-key",
                "api_secret": "demo-orion-secret",
                "capabilities": ["natural_language", "conversation", "reasoning"]
            },
            "sparknoca": {
                "endpoint_url": "https://api.sparknoca.analytics/v1",
                "websocket_url": "wss://ws.sparknoca.analytics/v1",
                "api_key": "demo-sparknoca-key",
                "api_secret": "demo-sparknoca-secret",
                "capabilities": ["analytics", "monitoring", "performance_tracking"]
            }
        }
        
        # Connect to each project
        for project_id, config in ecosystem_projects.items():
            try:
                logger.info(f"🔌 Connecting to {project_id}...")
                
                success = await self.connector.connect_to_project(project_id, config)
                
                if success:
                    self.connected_projects.append(project_id)
                    logger.info(f"✅ Connected to {project_id}")
                else:
                    logger.warning(f"⚠️  Failed to connect to {project_id}")
                    
            except Exception as e:
                logger.error(f"❌ Error connecting to {project_id}: {str(e)}")
        
        logger.info(f"🌐 Connected to {len(self.connected_projects)} ecosystem projects")
    
    async def create_ecosystem_agents(self):
        """Create agents that can work across the ecosystem"""
        
        logger.info("🤖 Creating ecosystem integration agents...")
        
        # Create mission coordinator agent
        mission_coordinator = MissionCoordinatorAgent(
            agent_id="mission-coordinator",
            capabilities=[AgentCapability.REASONING, AgentCapability.COORDINATION]
        )
        
        # Create system monitor agent
        system_monitor = SystemMonitorAgent(
            agent_id="system-monitor", 
            capabilities=[AgentCapability.MONITORING, AgentCapability.ANALYSIS]
        )
        
        # Create data integration agent
        data_integrator = DataIntegrationAgent(
            agent_id="data-integrator",
            capabilities=[AgentCapability.DATA_PROCESSING, AgentCapability.INTEGRATION]
        )
        
        self.agents = {
            "mission_coordinator": mission_coordinator,
            "system_monitor": system_monitor,
            "data_integrator": data_integrator
        }
        
        logger.info(f"✅ Created {len(self.agents)} ecosystem agents")
    
    async def demonstrate_multi_project_mission(self):
        """Demonstrate a mission that involves multiple ecosystem projects"""
        
        logger.info("🎯 Demonstrating multi-project ecosystem mission...")
        
        # Define a complex mission involving multiple projects
        mission_config = {
            "mission_id": "ecosystem-demo-001",
            "mission_type": "multi_domain_survey",
            "description": "Comprehensive survey combining space, ground, and analytical capabilities",
            "participants": ["nis-x", "nis-drone", "sparknoca"],
            "coordination_lead": "nis-hub",
            "duration": 3600,  # 1 hour
            "phases": [
                {
                    "phase_id": "orbital_survey",
                    "lead": "nis-x",
                    "duration": 1200,
                    "description": "Orbital reconnaissance and target identification",
                    "coordination_with": ["nis-hub", "sparknoca"]
                },
                {
                    "phase_id": "ground_survey", 
                    "lead": "nis-drone",
                    "duration": 1800,
                    "description": "Detailed ground survey using drone swarms",
                    "coordination_with": ["nis-x", "sparknoca"]
                },
                {
                    "phase_id": "data_analysis",
                    "lead": "sparknoca", 
                    "duration": 600,
                    "description": "Comprehensive data analysis and reporting",
                    "coordination_with": ["nis-x", "nis-drone", "orion"]
                }
            ]
        }
        
        # Coordinate the mission
        try:
            result = await self.connector.coordinate_mission(
                mission_config, 
                self.connected_projects
            )
            
            logger.info(f"✅ Mission coordination initiated: {result['mission_id']}")
            
            # Track mission progress
            await self._track_mission_progress(result['mission_id'])
            
        except Exception as e:
            logger.error(f"❌ Mission coordination failed: {str(e)}")
    
    async def demonstrate_data_integration(self):
        """Demonstrate data integration across ecosystem projects"""
        
        logger.info("📊 Demonstrating ecosystem data integration...")
        
        # Define data sources and targets
        data_integration_config = {
            "integration_id": "ecosystem-data-001",
            "sources": [
                {
                    "project": "nis-x",
                    "data_type": "orbital_telemetry",
                    "format": "json"
                },
                {
                    "project": "nis-drone", 
                    "data_type": "ground_sensor_data",
                    "format": "json"
                }
            ],
            "processing": {
                "coordinator": "sparknoca",
                "transformations": ["normalize", "correlate", "analyze"]
            },
            "targets": [
                {
                    "project": "orion",
                    "data_type": "processed_insights",
                    "format": "natural_language"
                },
                {
                    "project": "nis-hub",
                    "data_type": "mission_summary", 
                    "format": "structured"
                }
            ]
        }
        
        # Execute data integration
        try:
            result = await self.connector.sync_data_advanced(data_integration_config)
            
            logger.info(f"✅ Data integration completed: {result['integration_id']}")
            
            # Validate data integrity
            await self._validate_data_integration(result['integration_id'])
            
        except Exception as e:
            logger.error(f"❌ Data integration failed: {str(e)}")
    
    async def demonstrate_ecosystem_monitoring(self):
        """Demonstrate real-time ecosystem monitoring"""
        
        logger.info("📈 Demonstrating ecosystem monitoring...")
        
        # Get comprehensive ecosystem status
        ecosystem_status = await self.connector.get_integration_status()
        
        # Log ecosystem health
        logger.info("🌐 Ecosystem Status:")
        logger.info(f"  Connected Projects: {len(ecosystem_status['project_status'])}")
        logger.info(f"  Active Connections: {ecosystem_status['connection_statistics']['total_connections']}")
        logger.info(f"  Message Throughput: {ecosystem_status.get('message_throughput', 'N/A')}")
        
        # Monitor individual project health
        for project_id, status in ecosystem_status['project_status'].items():
            health_emoji = "🟢" if status['status'] == 'connected' else "🔴"
            logger.info(f"  {health_emoji} {project_id}: {status['status']}")
        
        # Check system health indicators
        system_health = ecosystem_status.get('system_health', {})
        for system, status in system_health.items():
            health_emoji = "🟢" if status == 'operational' else "🔴"
            logger.info(f"  {health_emoji} {system}: {status}")
    
    async def _track_mission_progress(self, mission_id: str):
        """Track the progress of a multi-project mission"""
        
        logger.info(f"📋 Tracking mission progress: {mission_id}")
        
        # Simulate mission progress tracking
        phases = ["orbital_survey", "ground_survey", "data_analysis"]
        
        for phase in phases:
            await asyncio.sleep(2)  # Simulate phase duration
            logger.info(f"  🔄 Phase '{phase}' in progress...")
            
            # Get status from relevant projects
            if phase == "orbital_survey":
                await self._get_project_status("nis-x")
            elif phase == "ground_survey":
                await self._get_project_status("nis-drone") 
            elif phase == "data_analysis":
                await self._get_project_status("sparknoca")
        
        logger.info(f"✅ Mission {mission_id} completed successfully")
    
    async def _get_project_status(self, project_id: str):
        """Get status from a specific project"""
        
        if project_id in self.connected_projects:
            # Simulate project status check
            logger.info(f"  📊 {project_id} status: Active, Performance: Optimal")
        else:
            logger.warning(f"  ⚠️  {project_id} not connected")
    
    async def _validate_data_integration(self, integration_id: str):
        """Validate data integration results"""
        
        logger.info(f"🔍 Validating data integration: {integration_id}")
        
        # Simulate data validation
        await asyncio.sleep(1)
        
        validation_results = {
            "data_consistency": "PASS",
            "format_compliance": "PASS", 
            "completeness": "PASS",
            "quality_score": 0.95
        }
        
        logger.info(f"✅ Data validation completed:")
        for metric, result in validation_results.items():
            logger.info(f"  📊 {metric}: {result}")
    
    async def cleanup_connections(self):
        """Clean up all ecosystem connections"""
        
        logger.info("🧹 Cleaning up ecosystem connections...")
        
        for project_id in self.connected_projects:
            try:
                await self.connector.disconnect_project(project_id)
                logger.info(f"✅ Disconnected from {project_id}")
            except Exception as e:
                logger.warning(f"⚠️  Error disconnecting from {project_id}: {str(e)}")
        
        logger.info("✅ Cleanup completed")


class MissionCoordinatorAgent(NISAgent):
    """Agent specialized in coordinating multi-project missions"""
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability]):
        super().__init__(agent_id, capabilities)
        self.coordination_history = []
    
    async def coordinate_mission(self, mission_config: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate a complex multi-project mission"""
        
        self.logger.info(f"🎯 Coordinating mission: {mission_config['mission_id']}")
        
        # Mission coordination logic
        coordination_result = {
            "mission_id": mission_config["mission_id"],
            "status": "coordinated",
            "participants": mission_config["participants"],
            "estimated_duration": mission_config["duration"],
            "coordination_timestamp": datetime.now().isoformat()
        }
        
        self.coordination_history.append(coordination_result)
        
        return coordination_result


class SystemMonitorAgent(NISAgent):
    """Agent specialized in monitoring ecosystem health"""
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability]):
        super().__init__(agent_id, capabilities)
        self.monitoring_data = []
    
    async def monitor_ecosystem_health(self, projects: List[str]) -> Dict[str, Any]:
        """Monitor the health of ecosystem projects"""
        
        self.logger.info("📊 Monitoring ecosystem health...")
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "projects_monitored": len(projects),
            "overall_health": "GOOD",
            "recommendations": []
        }
        
        self.monitoring_data.append(health_report)
        
        return health_report


class DataIntegrationAgent(NISAgent):
    """Agent specialized in cross-project data integration"""
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability]):
        super().__init__(agent_id, capabilities)
        self.integration_logs = []
    
    async def integrate_data(self, sources: List[Dict], targets: List[Dict]) -> Dict[str, Any]:
        """Integrate data across multiple ecosystem projects"""
        
        self.logger.info("🔗 Integrating data across ecosystem...")
        
        integration_result = {
            "integration_id": f"data-int-{int(datetime.now().timestamp())}",
            "sources_count": len(sources),
            "targets_count": len(targets), 
            "status": "completed",
            "data_volume": "2.5GB",
            "processing_time": "45s"
        }
        
        self.integration_logs.append(integration_result)
        
        return integration_result


async def main():
    """Main demonstration function"""
    
    print("🚀 NIS-TOOLKIT-SUIT Ecosystem Integration Demo")
    print("=" * 60)
    
    demo = EcosystemIntegrationDemo()
    
    try:
        # Initialize ecosystem connection
        await demo.initialize_ecosystem_connection()
        
        # Connect to all ecosystem projects
        await demo.connect_to_all_projects()
        
        # Create ecosystem agents
        await demo.create_ecosystem_agents()
        
        # Demonstrate ecosystem capabilities
        await demo.demonstrate_multi_project_mission()
        await demo.demonstrate_data_integration()
        await demo.demonstrate_ecosystem_monitoring()
        
        # Wait a moment to see all the logs
        await asyncio.sleep(2)
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        
    finally:
        # Clean up connections
        await demo.cleanup_connections()
    
    print("=" * 60)
    print("✅ Ecosystem Integration Demo Complete!")
    print("\n💡 This demo shows how NIS-TOOLKIT-SUIT enables:")
    print("  • Connection to multiple NIS projects")
    print("  • Multi-project mission coordination")
    print("  • Cross-system data integration")
    print("  • Real-time ecosystem monitoring")
    print("  • Unified agent deployment")
    print("\n🌐 Ready to build your own ecosystem integration!")


if __name__ == "__main__":
    # Run the ecosystem integration demo
    asyncio.run(main()) 