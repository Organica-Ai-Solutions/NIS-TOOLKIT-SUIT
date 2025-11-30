#!/usr/bin/env python3
"""
NIS Protocol v4.0 Integration Example

This example demonstrates how to use the NIS-TOOLKIT-SUIT with the new
NIS Protocol v4.0 features including:

- 10-Phase Consciousness Pipeline
- Robotics Integration (FK/IK, trajectory planning)
- Physics Validation (PINN)
- Authentication
- Flutter Desktop Client Configuration

Requirements:
- NIS Protocol v4.0 server running at http://localhost:8000
- Python 3.8+
- aiohttp, requests packages

Usage:
    python nis_v4_integration_example.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nis_core_toolkit.src.adapters import (
    NISv4Adapter,
    NISv4Config,
    ConsciousnessPhase,
    RoboticsCapability,
    create_nis_v4_adapter
)
from nis_core_toolkit.src.adapters.flutter_client_config import (
    FlutterClientConfig,
    FlutterClientManager,
    setup_flutter_integration
)


# ============================================================================
# Configuration
# ============================================================================

NIS_V4_URL = "http://localhost:8000"
API_KEY = None  # Set your API key here or use environment variable
JWT_TOKEN = None  # Or use JWT authentication


# ============================================================================
# Basic Connection Example
# ============================================================================

def example_basic_connection():
    """Demonstrate basic connection to NIS Protocol v4.0."""
    print("\n" + "="*60)
    print("Example 1: Basic Connection")
    print("="*60)
    
    # Create adapter with simple configuration
    adapter = create_nis_v4_adapter(
        base_url=NIS_V4_URL,
        api_key=API_KEY
    )
    
    # Check connection status
    status = adapter.get_status()
    print(f"\nAdapter Status:")
    print(f"  - Connection: {status['connection_status']}")
    print(f"  - Protocol: {status['protocol_name']}")
    print(f"  - WebSocket Enabled: {status['v4_config']['websocket_enabled']}")
    print(f"  - Robotics Enabled: {status['v4_config']['robotics_enabled']}")
    print(f"  - Consciousness Enabled: {status['v4_config']['consciousness_enabled']}")
    
    return adapter


# ============================================================================
# Consciousness Pipeline Example
# ============================================================================

async def example_consciousness_pipeline(adapter: NISv4Adapter):
    """Demonstrate the 10-phase consciousness pipeline."""
    print("\n" + "="*60)
    print("Example 2: 10-Phase Consciousness Pipeline")
    print("="*60)
    
    # Run individual phases
    print("\n--- Running Genesis Phase (Agent Generation) ---")
    try:
        genesis_result = adapter.consciousness_genesis(
            prompt="Design an autonomous drone delivery system for urban areas",
            capability="reasoning"
        )
        print(f"Genesis Output: {json.dumps(genesis_result, indent=2)[:500]}...")
    except Exception as e:
        print(f"Genesis phase error (server may not be running): {e}")
    
    print("\n--- Running Plan Phase (Strategic Planning) ---")
    try:
        plan_result = adapter.consciousness_plan(
            high_level_goal="Implement safe drone navigation in crowded urban environments",
            goal_id="drone_nav_001"
        )
        print(f"Plan Output: {json.dumps(plan_result, indent=2)[:500]}...")
    except Exception as e:
        print(f"Plan phase error: {e}")
    
    print("\n--- Running Ethics Phase (Ethical Evaluation) ---")
    try:
        ethics_result = adapter.consciousness_ethics(
            action="Deploy autonomous drones in residential neighborhoods",
            context={
                "stakeholders": ["residents", "businesses", "local government"],
                "potential_risks": ["privacy", "noise", "safety"]
            }
        )
        print(f"Ethics Output: {json.dumps(ethics_result, indent=2)[:500]}...")
    except Exception as e:
        print(f"Ethics phase error: {e}")
    
    # Run full pipeline (async)
    print("\n--- Running Full Consciousness Pipeline ---")
    try:
        full_result = await adapter.run_consciousness_pipeline(
            input_data="Create an AI assistant for scientific research",
            phases=[
                ConsciousnessPhase.GENESIS,
                ConsciousnessPhase.PLAN,
                ConsciousnessPhase.ETHICS,
                ConsciousnessPhase.REFLECTION
            ],
            context={"domain": "scientific_research", "priority": "accuracy"}
        )
        print(f"Pipeline completed in {full_result['metadata']['duration']:.2f}s")
        print(f"Phases executed: {full_result['metadata']['phases_executed']}")
    except Exception as e:
        print(f"Full pipeline error: {e}")


# ============================================================================
# Robotics Example
# ============================================================================

def example_robotics(adapter: NISv4Adapter):
    """Demonstrate robotics integration."""
    print("\n" + "="*60)
    print("Example 3: Robotics Integration")
    print("="*60)
    
    # Get robotics capabilities
    print("\n--- Checking Robotics Capabilities ---")
    try:
        capabilities = adapter.robotics_capabilities()
        print(f"Available capabilities: {json.dumps(capabilities, indent=2)}")
    except Exception as e:
        print(f"Capabilities check error: {e}")
    
    # Forward Kinematics
    print("\n--- Forward Kinematics ---")
    try:
        # Example: 6-DOF robot arm joint angles (in radians)
        joint_angles = [0.0, -0.5, 1.0, 0.0, 0.5, 0.0]
        
        fk_result = adapter.robotics_forward_kinematics(
            joint_angles=joint_angles,
            robot_config={
                "type": "6dof_arm",
                "dh_parameters": {
                    "a": [0, 0.4, 0.4, 0, 0, 0],
                    "d": [0.1, 0, 0, 0.1, 0, 0.05],
                    "alpha": [-1.5708, 0, 0, -1.5708, 1.5708, 0],
                    "theta_offset": [0, 0, 0, 0, 0, 0]
                }
            }
        )
        print(f"End Effector Pose: {json.dumps(fk_result, indent=2)}")
    except Exception as e:
        print(f"Forward kinematics error: {e}")
    
    # Inverse Kinematics
    print("\n--- Inverse Kinematics ---")
    try:
        target_pose = {
            "position": {"x": 0.5, "y": 0.2, "z": 0.3},
            "orientation": {"roll": 0, "pitch": 0, "yaw": 0}
        }
        
        ik_result = adapter.robotics_inverse_kinematics(
            target_pose=target_pose,
            constraints={
                "joint_limits": [
                    {"min": -3.14, "max": 3.14},
                    {"min": -1.57, "max": 1.57},
                    {"min": -2.0, "max": 2.0},
                    {"min": -3.14, "max": 3.14},
                    {"min": -1.57, "max": 1.57},
                    {"min": -3.14, "max": 3.14}
                ]
            }
        )
        print(f"Joint Angles Solution: {json.dumps(ik_result, indent=2)}")
    except Exception as e:
        print(f"Inverse kinematics error: {e}")
    
    # Trajectory Planning
    print("\n--- Trajectory Planning ---")
    try:
        trajectory = adapter.robotics_plan_trajectory(
            start_pose={
                "position": {"x": 0.3, "y": 0.0, "z": 0.4},
                "orientation": {"roll": 0, "pitch": 0, "yaw": 0}
            },
            end_pose={
                "position": {"x": 0.5, "y": 0.3, "z": 0.2},
                "orientation": {"roll": 0, "pitch": 0.5, "yaw": 0.3}
            },
            waypoints=[
                {"position": {"x": 0.4, "y": 0.15, "z": 0.35}}
            ],
            constraints={
                "max_velocity": 1.0,
                "max_acceleration": 0.5,
                "collision_check": True
            }
        )
        print(f"Trajectory Points: {len(trajectory.get('trajectory', []))} waypoints")
    except Exception as e:
        print(f"Trajectory planning error: {e}")


# ============================================================================
# Physics Example
# ============================================================================

def example_physics(adapter: NISv4Adapter):
    """Demonstrate physics validation with PINN."""
    print("\n" + "="*60)
    print("Example 4: Physics Validation (PINN)")
    print("="*60)
    
    # Get physical constants
    print("\n--- Physical Constants ---")
    try:
        constants = adapter.physics_constants()
        print(f"Available constants: {list(constants.keys())[:10]}...")
    except Exception as e:
        print(f"Constants error: {e}")
    
    # Validate physics constraints
    print("\n--- Physics Constraint Validation ---")
    try:
        validation = adapter.physics_validate({
            "conservation_laws": ["energy", "momentum"],
            "system": {
                "type": "mechanical",
                "components": ["mass", "spring", "damper"],
                "initial_energy": 100.0,
                "final_energy": 95.0,  # Some energy loss due to damping
                "energy_dissipated": 5.0
            }
        })
        print(f"Validation Result: {json.dumps(validation, indent=2)}")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Solve heat equation
    print("\n--- Heat Equation Solver (PINN) ---")
    try:
        heat_solution = adapter.physics_solve_heat_equation(
            parameters={
                "thermal_diffusivity": 0.01,
                "domain": {"x_min": 0, "x_max": 1, "t_min": 0, "t_max": 1}
            },
            boundary_conditions={
                "left": {"type": "dirichlet", "value": 100},
                "right": {"type": "dirichlet", "value": 0}
            },
            initial_conditions={
                "type": "uniform",
                "value": 50
            }
        )
        print(f"Heat Solution: {json.dumps(heat_solution, indent=2)[:500]}...")
    except Exception as e:
        print(f"Heat equation error: {e}")


# ============================================================================
# Authentication Example
# ============================================================================

def example_authentication(adapter: NISv4Adapter):
    """Demonstrate authentication flow."""
    print("\n" + "="*60)
    print("Example 5: Authentication")
    print("="*60)
    
    # Note: These will fail without valid credentials
    # This is just to demonstrate the API
    
    print("\n--- Login (Demo) ---")
    try:
        # In production, use secure credential handling
        login_result = adapter.auth_login(
            username="demo_user",
            password="demo_password"
        )
        print(f"Login successful: {login_result.get('authenticated', False)}")
    except Exception as e:
        print(f"Login error (expected without valid credentials): {e}")
    
    print("\n--- Verify Token ---")
    try:
        verify_result = adapter.auth_verify()
        print(f"Token valid: {verify_result}")
    except Exception as e:
        print(f"Verify error: {e}")


# ============================================================================
# Flutter Client Configuration Example
# ============================================================================

def example_flutter_config():
    """Demonstrate Flutter client configuration generation."""
    print("\n" + "="*60)
    print("Example 6: Flutter Desktop Client Configuration")
    print("="*60)
    
    # Create Flutter configuration
    flutter_config = FlutterClientConfig(
        api_base_url=NIS_V4_URL,
        websocket_url=f"ws://localhost:8000/ws/agentic",
        enable_agentic_chat=True,
        enable_telemetry=True,
        enable_robotics_panel=True,
        enable_bitnet_monitor=True,
        enable_consciousness_viewer=True,
        theme="dark"
    )
    
    # Create manager
    manager = FlutterClientManager(flutter_config)
    
    # Generate complete configuration
    full_config = manager.generate_client_config()
    print("\n--- Generated Flutter Configuration ---")
    print(f"Version: {full_config['version']}")
    print(f"Features enabled: {sum(full_config['features'].values())}/{len(full_config['features'])}")
    print(f"Endpoints configured: {len(full_config['endpoints'])}")
    
    # Show feature flags
    print("\n--- Feature Flags ---")
    for feature, enabled in full_config['features'].items():
        status = "✓" if enabled else "✗"
        print(f"  [{status}] {feature}")
    
    # Generate Dart code
    print("\n--- Generated Dart Configuration Class ---")
    dart_code = manager.generate_dart_config_class()
    print(dart_code[:1000] + "...")
    
    # Export configuration files
    print("\n--- Exporting Configuration Files ---")
    try:
        output_dir = Path(__file__).parent / "flutter_output"
        files = setup_flutter_integration(
            api_url=NIS_V4_URL,
            output_dir=str(output_dir)
        )
        print(f"JSON Config: {files['json_config']}")
        print(f"Dart Config: {files['dart_config']}")
    except Exception as e:
        print(f"Export error: {e}")


# ============================================================================
# WebSocket Streaming Example
# ============================================================================

async def example_websocket_streaming(adapter: NISv4Adapter):
    """Demonstrate WebSocket streaming for real-time communication."""
    print("\n" + "="*60)
    print("Example 7: WebSocket Streaming")
    print("="*60)
    
    print("\n--- Connecting to WebSocket ---")
    try:
        await adapter.connect_websocket("/ws/agentic")
        print("WebSocket connected!")
        
        # Send a message
        await adapter.send_websocket_message({
            "type": "chat",
            "content": "Hello from NIS-TOOLKIT-SUIT!",
            "agent": "consciousness"
        })
        print("Message sent!")
        
        # Receive response (with timeout)
        try:
            response = await asyncio.wait_for(
                adapter.receive_websocket_message(),
                timeout=5.0
            )
            print(f"Received: {json.dumps(response, indent=2)}")
        except asyncio.TimeoutError:
            print("No response received (timeout)")
        
        # Close connection
        await adapter.close_websocket()
        print("WebSocket closed.")
        
    except Exception as e:
        print(f"WebSocket error (server may not be running): {e}")


# ============================================================================
# Message Translation Example
# ============================================================================

def example_message_translation(adapter: NISv4Adapter):
    """Demonstrate message translation between formats."""
    print("\n" + "="*60)
    print("Example 8: Message Translation")
    print("="*60)
    
    # Translate NIS toolkit message to v4 format
    print("\n--- NIS Toolkit → v4 Format ---")
    nis_message = {
        "message_type": "consciousness",
        "payload": {
            "phase": "genesis",
            "input": "Design a smart home system",
            "context": {"domain": "iot"}
        }
    }
    v4_message = adapter.translate_from_nis(nis_message)
    print(f"Original: {json.dumps(nis_message, indent=2)}")
    print(f"Translated: {json.dumps(v4_message, indent=2)}")
    
    # Translate v4 response to NIS toolkit format
    print("\n--- v4 Response → NIS Toolkit Format ---")
    v4_response = {
        "_endpoint": "/v4/consciousness/genesis",
        "phase": "genesis",
        "output": {
            "ideas": ["Smart lighting", "Climate control", "Security system"],
            "confidence": 0.95
        },
        "reasoning": ["Analyzed user requirements", "Generated innovative solutions"]
    }
    nis_response = adapter.translate_to_nis(v4_response)
    print(f"Original: {json.dumps(v4_response, indent=2)}")
    print(f"Translated: {json.dumps(nis_response, indent=2)}")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("NIS Protocol v4.0 Integration Examples")
    print("NIS-TOOLKIT-SUIT")
    print("="*60)
    
    # Example 1: Basic Connection
    adapter = example_basic_connection()
    
    # Example 2: Consciousness Pipeline
    await example_consciousness_pipeline(adapter)
    
    # Example 3: Robotics
    example_robotics(adapter)
    
    # Example 4: Physics
    example_physics(adapter)
    
    # Example 5: Authentication
    example_authentication(adapter)
    
    # Example 6: Flutter Configuration
    example_flutter_config()
    
    # Example 7: WebSocket Streaming
    await example_websocket_streaming(adapter)
    
    # Example 8: Message Translation
    example_message_translation(adapter)
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)
    print("\nNote: Some examples may show errors if the NIS Protocol v4.0")
    print("server is not running at", NIS_V4_URL)


if __name__ == "__main__":
    asyncio.run(main())
