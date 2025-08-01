#!/usr/bin/env python3
"""
NIS Integration CLI - Real Project Connections
Command-line interface for connecting to actual NIS projects
"""

import asyncio
import click
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import yaml
import os
from datetime import datetime

# Import the integration connector
try:
    from ..core.integration_connector import create_nis_integration_connector, MessageType
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from core.integration_connector import create_nis_integration_connector, MessageType

class NISConnectionManager:
    """Manages NIS project connections"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or os.path.expanduser("~/.nis/connections.yaml")
        self.config_dir = Path(self.config_file).parent
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.connections = self._load_connections()
        self.connector = None
    
    def _load_connections(self) -> Dict[str, Any]:
        """Load saved connections"""
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _save_connections(self):
        """Save connections to file"""
        with open(self.config_file, 'w') as f:
            yaml.dump(self.connections, f, default_flow_style=False)
    
    def add_connection(self, project_id: str, connection_config: Dict[str, Any]):
        """Add connection configuration"""
        self.connections[project_id] = {
            **connection_config,
            "added_at": datetime.now().isoformat(),
            "last_used": None
        }
        self._save_connections()
    
    def remove_connection(self, project_id: str):
        """Remove connection configuration"""
        if project_id in self.connections:
            del self.connections[project_id]
            self._save_connections()
    
    def get_connection(self, project_id: str) -> Dict[str, Any]:
        """Get connection configuration"""
        return self.connections.get(project_id, {})
    
    def list_connections(self) -> List[str]:
        """List all saved connections"""
        return list(self.connections.keys())
    
    async def connect_to_project(self, project_id: str, test_only: bool = False) -> bool:
        """Connect to a project"""
        if not self.connector:
            self.connector = create_nis_integration_connector(
                connector_id="NIS-TOOLKIT-CLI",
                config={"authentication_enabled": True}
            )
        
        connection_config = self.get_connection(project_id)
        if not connection_config:
            click.echo(f"❌ No connection configuration found for {project_id}")
            return False
        
        try:
            success = await self.connector.connect_to_project(project_id, connection_config)
            
            if success:
                # Update last used
                self.connections[project_id]["last_used"] = datetime.now().isoformat()
                self._save_connections()
                
                if test_only:
                    await self.connector.disconnect_project(project_id)
                
                return True
            else:
                return False
                
        except Exception as e:
            click.echo(f"❌ Connection failed: {str(e)}")
            return False

# Initialize connection manager
connection_manager = NISConnectionManager()

@click.group()
def connect():
    """Connect to NIS projects"""
    pass

@connect.command()
@click.argument('project_id', type=click.Choice([
    'nis-x', 'nis-drone', 'archaeological-research', 
    'sparknova', 'orion', 'sparknoca', 'nis-hub'
]))
@click.option('--endpoint', help='API endpoint URL')
@click.option('--websocket', help='WebSocket URL')
@click.option('--api-key', help='API key for authentication')
@click.option('--api-secret', help='API secret for authentication')
@click.option('--config-file', help='Load configuration from file')
@click.option('--test', is_flag=True, help='Test connection without saving')
def add(project_id, endpoint, websocket, api_key, api_secret, config_file, test):
    """Add connection to NIS project"""
    
    if config_file:
        # Load from file
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Build from command line options
        config = {}
        if endpoint:
            config['endpoint_url'] = endpoint
        if websocket:
            config['websocket_url'] = websocket
        if api_key:
            config['api_key'] = api_key
        if api_secret:
            config['api_secret'] = api_secret
    
    if not config:
        click.echo("❌ No configuration provided. Use --config-file or provide connection details.")
        return
    
    if not test:
        # Save connection
        connection_manager.add_connection(project_id, config)
        click.echo(f"✅ Connection configuration saved for {project_id}")
    
    # Test connection
    click.echo(f"🔗 Testing connection to {project_id}...")
    
    async def test_connection():
        success = await connection_manager.connect_to_project(project_id, test_only=True)
        if success:
            click.echo(f"✅ Connection to {project_id} successful!")
        else:
            click.echo(f"❌ Connection to {project_id} failed!")
    
    asyncio.run(test_connection())

@connect.command()
@click.argument('project_id', required=False)
def list(project_id):
    """List connections or show connection details"""
    
    if project_id:
        # Show specific connection
        config = connection_manager.get_connection(project_id)
        if config:
            click.echo(f"📋 Connection details for {project_id}:")
            click.echo(f"  Endpoint: {config.get('endpoint_url', 'Not set')}")
            click.echo(f"  WebSocket: {config.get('websocket_url', 'Not set')}")
            click.echo(f"  API Key: {'Set' if config.get('api_key') else 'Not set'}")
            click.echo(f"  Added: {config.get('added_at', 'Unknown')}")
            click.echo(f"  Last Used: {config.get('last_used', 'Never')}")
        else:
            click.echo(f"❌ No connection found for {project_id}")
    else:
        # List all connections
        connections = connection_manager.list_connections()
        if connections:
            click.echo("📋 Saved connections:")
            for conn in connections:
                config = connection_manager.get_connection(conn)
                last_used = config.get('last_used', 'Never')
                click.echo(f"  • {conn} (Last used: {last_used})")
        else:
            click.echo("📋 No saved connections")

@connect.command()
@click.argument('project_id')
def remove(project_id):
    """Remove connection configuration"""
    
    if project_id in connection_manager.list_connections():
        connection_manager.remove_connection(project_id)
        click.echo(f"✅ Connection configuration removed for {project_id}")
    else:
        click.echo(f"❌ No connection found for {project_id}")

@connect.command()
@click.argument('project_id')
@click.option('--timeout', default=30, help='Connection timeout in seconds')
def test(project_id, timeout):
    """Test connection to NIS project"""
    
    click.echo(f"🔗 Testing connection to {project_id}...")
    
    async def test_connection():
        try:
            success = await connection_manager.connect_to_project(project_id, test_only=True)
            if success:
                click.echo(f"✅ Connection to {project_id} successful!")
            else:
                click.echo(f"❌ Connection to {project_id} failed!")
        except Exception as e:
            click.echo(f"❌ Connection error: {str(e)}")
    
    asyncio.run(test_connection())

@connect.command()
@click.argument('project_ids', nargs=-1, required=True)
@click.option('--mission-id', help='Mission ID for coordination')
@click.option('--mission-type', default='coordination', help='Type of mission')
@click.option('--duration', default=3600, help='Mission duration in seconds')
def coordinate(project_ids, mission_id, mission_type, duration):
    """Coordinate mission across multiple NIS projects"""
    
    if not mission_id:
        mission_id = f"mission_{int(datetime.now().timestamp())}"
    
    click.echo(f"🚀 Coordinating mission {mission_id} across {len(project_ids)} projects...")
    
    async def coordinate_mission():
        try:
            # Initialize connector
            connector = create_nis_integration_connector(
                connector_id="NIS-TOOLKIT-CLI",
                config={"authentication_enabled": True}
            )
            
            # Connect to all projects
            connected_projects = []
            for project_id in project_ids:
                config = connection_manager.get_connection(project_id)
                if config:
                    success = await connector.connect_to_project(project_id, config)
                    if success:
                        connected_projects.append(project_id)
                        click.echo(f"  ✅ Connected to {project_id}")
                    else:
                        click.echo(f"  ❌ Failed to connect to {project_id}")
                else:
                    click.echo(f"  ❌ No configuration for {project_id}")
            
            if not connected_projects:
                click.echo("❌ No successful connections. Cannot coordinate mission.")
                return
            
            # Coordinate mission
            mission_data = {
                "mission_id": mission_id,
                "mission_type": mission_type,
                "duration": duration,
                "parameters": {
                    "start_time": datetime.now().isoformat(),
                    "participating_projects": connected_projects
                }
            }
            
            result = await connector.coordinate_mission(mission_data, connected_projects)
            
            click.echo(f"✅ Mission coordination initiated!")
            click.echo(f"  Mission ID: {result['mission_id']}")
            click.echo(f"  Participating projects: {', '.join(result['participating_projects'])}")
            click.echo(f"  Coordination time: {result['coordination_timestamp']}")
            
        except Exception as e:
            click.echo(f"❌ Mission coordination failed: {str(e)}")
    
    asyncio.run(coordinate_mission())

@connect.command()
@click.argument('source_project')
@click.argument('target_projects', nargs=-1, required=True)
@click.option('--data-type', default='general', help='Type of data to sync')
@click.option('--data-file', help='File containing data to sync')
@click.option('--data-json', help='JSON data to sync')
def sync(source_project, target_projects, data_type, data_file, data_json):
    """Sync data between NIS projects"""
    
    # Prepare data payload
    data_payload = {}
    if data_file:
        with open(data_file, 'r') as f:
            if data_file.endswith('.json'):
                data_payload = json.load(f)
            elif data_file.endswith('.yaml') or data_file.endswith('.yml'):
                data_payload = yaml.safe_load(f)
            else:
                data_payload = {"content": f.read()}
    elif data_json:
        data_payload = json.loads(data_json)
    else:
        data_payload = {"timestamp": datetime.now().isoformat()}
    
    click.echo(f"📊 Syncing {data_type} data from {source_project} to {len(target_projects)} projects...")
    
    async def sync_data():
        try:
            # Initialize connector
            connector = create_nis_integration_connector(
                connector_id="NIS-TOOLKIT-CLI",
                config={"authentication_enabled": True}
            )
            
            # Connect to all projects
            all_projects = [source_project] + list(target_projects)
            connected_projects = []
            
            for project_id in all_projects:
                config = connection_manager.get_connection(project_id)
                if config:
                    success = await connector.connect_to_project(project_id, config)
                    if success:
                        connected_projects.append(project_id)
                        click.echo(f"  ✅ Connected to {project_id}")
                    else:
                        click.echo(f"  ❌ Failed to connect to {project_id}")
                else:
                    click.echo(f"  ❌ No configuration for {project_id}")
            
            # Sync data
            sync_targets = [p for p in target_projects if p in connected_projects]
            if sync_targets:
                result = await connector.sync_data(
                    source_project=source_project,
                    target_projects=sync_targets,
                    data_type=data_type,
                    data_payload=data_payload
                )
                
                click.echo(f"✅ Data sync initiated!")
                click.echo(f"  Source: {result['source_project']}")
                click.echo(f"  Data type: {result['data_type']}")
                click.echo(f"  Target projects: {', '.join(sync_targets)}")
                click.echo(f"  Sync time: {result['sync_timestamp']}")
            else:
                click.echo("❌ No valid target projects for sync")
                
        except Exception as e:
            click.echo(f"❌ Data sync failed: {str(e)}")
    
    asyncio.run(sync_data())

@connect.command()
@click.option('--output-format', default='table', type=click.Choice(['table', 'json', 'yaml']))
def status(output_format):
    """Show integration status"""
    
    click.echo("📊 Getting integration status...")
    
    async def get_status():
        try:
            # Initialize connector
            connector = create_nis_integration_connector(
                connector_id="NIS-TOOLKIT-CLI",
                config={"authentication_enabled": True}
            )
            
            # Connect to all saved projects
            connections = connection_manager.list_connections()
            for project_id in connections:
                config = connection_manager.get_connection(project_id)
                try:
                    await connector.connect_to_project(project_id, config)
                except:
                    pass  # Continue even if connection fails
            
            # Get status
            status = await connector.get_integration_status()
            
            if output_format == 'json':
                click.echo(json.dumps(status, indent=2))
            elif output_format == 'yaml':
                click.echo(yaml.dump(status, default_flow_style=False))
            else:
                # Table format
                click.echo(f"📊 Integration Status")
                click.echo(f"  Connector ID: {status['connector_id']}")
                click.echo(f"  Timestamp: {status['timestamp']}")
                click.echo(f"  Total connections: {status['connection_statistics']['total_connections']}")
                click.echo(f"  WebSocket connections: {status['connection_statistics']['websocket_connections']}")
                click.echo(f"  Message handlers: {status['connection_statistics']['message_handlers']}")
                
                if status['project_status']:
                    click.echo(f"  📋 Project Status:")
                    for project_id, project_status in status['project_status'].items():
                        status_icon = "✅" if project_status['status'] == 'connected' else "❌"
                        click.echo(f"    {status_icon} {project_id}: {project_status['status']}")
                        if project_status['last_heartbeat']:
                            click.echo(f"      Last heartbeat: {project_status['last_heartbeat']}")
                        click.echo(f"      Capabilities: {', '.join(project_status['capabilities'])}")
                
        except Exception as e:
            click.echo(f"❌ Status check failed: {str(e)}")
    
    asyncio.run(get_status())

@connect.command()
def setup():
    """Setup NIS integration with example configurations"""
    
    click.echo("🔧 Setting up NIS integration...")
    
    example_configs = {
        "nis-x": {
            "endpoint_url": "https://api.nis-x.space/v1",
            "websocket_url": "wss://ws.nis-x.space/v1",
            "api_key": "your-nis-x-api-key",
            "api_secret": "your-nis-x-api-secret"
        },
        "nis-drone": {
            "endpoint_url": "https://api.nis-drone.com/v1",
            "websocket_url": "wss://ws.nis-drone.com/v1",
            "api_key": "your-nis-drone-api-key",
            "api_secret": "your-nis-drone-api-secret"
        },
        "archaeological-research": {
            "endpoint_url": "https://api.archaeological-research.org/v1",
            "websocket_url": "wss://ws.archaeological-research.org/v1",
            "api_key": "your-archaeological-api-key",
            "api_secret": "your-archaeological-api-secret"
        }
    }
    
    # Create example configuration file
    example_file = Path(connection_manager.config_dir) / "example_connections.yaml"
    with open(example_file, 'w') as f:
        yaml.dump(example_configs, f, default_flow_style=False)
    
    click.echo(f"✅ Example configuration created at: {example_file}")
    click.echo(f"📋 To use:")
    click.echo(f"  1. Edit {example_file} with your actual API credentials")
    click.echo(f"  2. Run: nis connect add nis-x --config-file {example_file}")
    click.echo(f"  3. Test: nis connect test nis-x")

if __name__ == '__main__':
    connect() 