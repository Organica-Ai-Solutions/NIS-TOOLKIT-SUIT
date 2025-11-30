# NIS Protocol Adapters for External Protocol Integration

This module provides adapters for integrating NIS Protocol with external AI agent communication protocols.

## Overview

The NIS Protocol can act as a meta-protocol, orchestrating and coordinating agents from different protocol ecosystems. The adapters in this module allow NIS Protocol to seamlessly communicate with:

- **NIS Protocol v4.0** - Full integration with the 10-phase consciousness pipeline, robotics, physics validation, and Flutter desktop client
- **MCP (Model Context Protocol)** - Anthropic's protocol for connecting AI systems to data sources
- **ACP (Agent Communication Protocol)** - IBM's standardized protocol for agent communication
- **A2A (Agent2Agent Protocol)** - Google's protocol for agent interoperability across platforms

## NIS Protocol v4.0 Integration (NEW)

The `NISv4Adapter` provides complete integration with NIS Protocol v4.0, including:

### 10-Phase Consciousness Pipeline

```python
from src.adapters import NISv4Adapter, ConsciousnessPhase, create_nis_v4_adapter

# Create adapter
adapter = create_nis_v4_adapter(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Run individual consciousness phases
genesis_result = adapter.consciousness_genesis(
    prompt="Design an autonomous delivery system"
)

plan_result = adapter.consciousness_plan(
    goal="Implement safe navigation",
    constraints={"max_speed": 10}
)

ethics_result = adapter.consciousness_ethics(
    action="Deploy autonomous drones",
    context={"stakeholders": ["residents", "businesses"]}
)

# Run full pipeline (async)
import asyncio

async def run_pipeline():
    result = await adapter.run_consciousness_pipeline(
        input_data="Create an AI research assistant",
        phases=[
            ConsciousnessPhase.GENESIS,
            ConsciousnessPhase.PLAN,
            ConsciousnessPhase.ETHICS,
            ConsciousnessPhase.REFLECTION
        ]
    )
    return result

result = asyncio.run(run_pipeline())
```

### Robotics Integration

```python
# Forward Kinematics
fk_result = adapter.robotics_forward_kinematics(
    joint_angles=[0.0, -0.5, 1.0, 0.0, 0.5, 0.0],
    robot_config={"type": "6dof_arm"}
)

# Inverse Kinematics
ik_result = adapter.robotics_inverse_kinematics(
    target_pose={
        "position": {"x": 0.5, "y": 0.2, "z": 0.3},
        "orientation": {"roll": 0, "pitch": 0, "yaw": 0}
    }
)

# Trajectory Planning
trajectory = adapter.robotics_plan_trajectory(
    start_pose={"position": {"x": 0.3, "y": 0.0, "z": 0.4}},
    end_pose={"position": {"x": 0.5, "y": 0.3, "z": 0.2}},
    constraints={"max_velocity": 1.0}
)
```

### Physics Validation (PINN)

```python
# Validate physics constraints
validation = adapter.physics_validate({
    "conservation_laws": ["energy", "momentum"],
    "system": {"type": "mechanical"}
})

# Solve heat equation
solution = adapter.physics_solve_heat_equation(
    parameters={"thermal_diffusivity": 0.01},
    boundary_conditions={"left": {"type": "dirichlet", "value": 100}},
    initial_conditions={"type": "uniform", "value": 50}
)
```

### Flutter Desktop Client Configuration

```python
from src.adapters import (
    FlutterClientConfig,
    FlutterClientManager,
    setup_flutter_integration
)

# Quick setup
files = setup_flutter_integration(
    api_url="http://localhost:8000",
    output_dir="./flutter_config"
)

# Or detailed configuration
config = FlutterClientConfig(
    api_base_url="http://localhost:8000",
    enable_agentic_chat=True,
    enable_robotics_panel=True,
    enable_consciousness_viewer=True
)

manager = FlutterClientManager(config)
manager.export_config_for_flutter("config.json")
manager.export_dart_config("nis_config.dart")
```

### WebSocket Streaming

```python
import asyncio

async def stream_example():
    await adapter.connect_websocket("/ws/agentic")
    
    await adapter.send_websocket_message({
        "type": "chat",
        "content": "Hello!",
        "agent": "consciousness"
    })
    
    response = await adapter.receive_websocket_message()
    print(response)
    
    await adapter.close_websocket()

asyncio.run(stream_example())
```

## NeuroLinux Integration (NEW)

The `NeuroLinuxAdapter` provides integration with NeuroLinux, the cognitive operating system for edge robotics and distributed AI.

### NeuroLinux Components

- **NeuroGrid**: Distributed mesh network for device coordination
- **NeuroHub**: Cloud controller for system management
- **NeuroKernel**: Rust-based agent scheduling kernel
- **NeuroForge**: SDK and development tools
- **NeuroStore**: Package registry for AI components

### Quick Start

```python
from src.adapters import create_neurolinux_adapter, neurolinux_quick_connect
import asyncio

# Quick connect
async def main():
    adapter = await neurolinux_quick_connect("http://localhost:8080")
    if adapter:
        print("Connected to NeuroLinux!")
        
        # Discover services
        services = await adapter.discover_services()
        print(f"Found {len(services)} services")
        
        # Deploy an agent to edge
        result = await adapter.deploy_agent(
            agent_id="consciousness_001",
            agent_type="consciousness",
            config={"model": "claude-3"},
            target_device="edge_device_001"
        )

asyncio.run(main())
```

### Service Management

```python
# Start a service
await adapter.start_service("nis_protocol", port=8000)

# Stop a service
await adapter.stop_service("nis_protocol")

# Get service status
status = await adapter.get_service_status("nis_protocol")
```

### Edge Device Management

```python
# Discover devices on NeuroGrid
devices = await adapter.discover_devices()

# Get device telemetry
telemetry = await adapter.get_device_telemetry("edge_001")

# Execute command on device
result = await adapter.execute_on_device(
    device_id="edge_001",
    command="neuroctl",
    args=["status"]
)
```

### NeuroKernel Task Scheduling

```python
# Schedule a task on the Rust NeuroKernel
result = await adapter.schedule_task(
    task_id="task_001",
    task_type="inference",
    priority=8,
    deadline_ms=100,
    payload={"model": "vision", "input": data}
)
```

### OTA Updates

```python
# Check for updates
updates = await adapter.check_updates()

# Deploy model update to devices
await adapter.deploy_model_update(
    model_id="consciousness_v2",
    version="2.0.0",
    target_devices=["edge_001", "edge_002"]
)
```

### Real-time Telemetry Streaming

```python
def handle_telemetry(data):
    print(f"CPU: {data['cpu']}%, Memory: {data['memory']}%")

await adapter.stream_telemetry(handle_telemetry, device_id="edge_001")
```

## Architecture

The adapter system consists of:

1. **BaseProtocolAdapter** - Abstract base class that all adapters implement
2. **CoordinatorAgent** - Central hub that manages message routing and translation
3. **Protocol-specific adapters** - Translate between NIS Protocol and external formats

```
┌─────────────────────────────────────┐
│          NIS Protocol Core          │
│                                     │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │ Vision  │ │ Memory  │ │  Other │ │
│  │ Agent   │ │ Agent   │ │ Agents │ │
│  └─────────┘ └─────────┘ └────────┘ │
│         │         │          │      │
│         └────┬────┴──────────┘      │
│              │                      │
│     ┌────────▼─────────┐            │
│     │  Coordinator     │            │
│     │     Agent        │            │
│     └────────┬─────────┘            │
└──────────────┼──────────────────────┘
               │
    ┌───────────────────────┐
    │   Protocol Adapters   │
    │                       │
┌───▼───┐   ┌───────┐   ┌───▼───┐
│  MCP  │   │  ACP  │   │  A2A  │
│Adapter│   │Adapter│   │Adapter│
└───┬───┘   └───┬───┘   └───┬───┘
    │           │           │
    ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│   MCP   │ │   ACP   │ │   A2A   │
│  Tools  │ │  Agents │ │  Agents │
└─────────┘ └─────────┘ └─────────┘
```

## Usage

### Basic Setup

To configure a NIS Protocol system with external protocol adapters:

```python
from src.agents.coordination.coordinator_agent import CoordinatorAgent
from src.adapters.bootstrap import configure_coordinator_agent

# Create coordinator agent
coordinator = CoordinatorAgent()

# Configure with adapters
configure_coordinator_agent(coordinator, config_path="path/to/config.json")
```

### Configuration

The adapter configuration is specified in a JSON file:

```json
{
  "mcp": {
    "base_url": "https://api.example.com/mcp",
    "api_key": "YOUR_MCP_API_KEY",
    "tool_mappings": {
      "vision_tool": {
        "nis_agent": "vision_agent",
        "target_layer": "PERCEPTION"
      }
    }
  },
  "acp": {
    "base_url": "https://api.example.com/acp",
    "api_key": "YOUR_ACP_API_KEY"
  },
  "a2a": {
    "base_url": "https://api.example.com/a2a",
    "api_key": "YOUR_A2A_API_KEY"
  }
}
```

### Processing External Protocol Messages

To process a message from an external protocol:

```python
# Process MCP message
mcp_response = coordinator.process({
    "protocol": "mcp",
    "original_message": mcp_message
})

# Process ACP message
acp_response = coordinator.process({
    "protocol": "acp",
    "original_message": acp_message
})

# Process A2A message
a2a_response = coordinator.process({
    "protocol": "a2a",
    "original_message": a2a_message
})
```

### Sending Messages to External Agents

To send a message to an external agent:

```python
# Send to MCP tool
mcp_response = coordinator.route_to_external_agent(
    "mcp",
    "vision_tool",
    nis_message
)

# Send to ACP agent
acp_response = coordinator.route_to_external_agent(
    "acp", 
    "factory_control_agent",
    nis_message
)

# Send to A2A agent
a2a_response = coordinator.route_to_external_agent(
    "a2a",
    "natural_language_agent",
    nis_message
)
```

## Adding New Protocol Adapters

To add support for a new protocol:

1. Create a new adapter class inheriting from `BaseProtocolAdapter`
2. Implement the required methods:
   - `translate_to_nis` - Convert from external format to NIS Protocol
   - `translate_from_nis` - Convert from NIS Protocol to external format
   - `send_to_external_agent` - Send messages to external agents

## Example

See the `examples/protocol_integration` directory for a full example demonstrating integration with all supported protocols. 