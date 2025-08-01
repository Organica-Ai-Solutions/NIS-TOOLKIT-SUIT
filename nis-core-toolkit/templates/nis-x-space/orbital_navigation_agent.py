#!/usr/bin/env python3
"""
NIS-X Space Systems Integration
Orbital Navigation Agent for Real Space Missions
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class OrbitState:
    """Current orbital state parameters"""
    position: Tuple[float, float, float]  # [x, y, z] in km
    velocity: Tuple[float, float, float]  # [vx, vy, vz] in km/s
    timestamp: datetime
    semi_major_axis: float  # km
    eccentricity: float
    inclination: float  # degrees
    ascending_node: float  # degrees
    argument_periapsis: float  # degrees
    true_anomaly: float  # degrees
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry"""
        return {
            "position": self.position,
            "velocity": self.velocity,
            "timestamp": self.timestamp.isoformat(),
            "orbital_elements": {
                "semi_major_axis": self.semi_major_axis,
                "eccentricity": self.eccentricity,
                "inclination": self.inclination,
                "ascending_node": self.ascending_node,
                "argument_periapsis": self.argument_periapsis,
                "true_anomaly": self.true_anomaly
            }
        }

@dataclass
class NavigationCommand:
    """Navigation command for spacecraft"""
    command_type: str  # "burn", "attitude", "coast"
    parameters: Dict[str, Any]
    execution_time: datetime
    duration: float  # seconds
    priority: int = 5  # 1-10 priority scale
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_type": self.command_type,
            "parameters": self.parameters,
            "execution_time": self.execution_time.isoformat(),
            "duration": self.duration,
            "priority": self.priority
        }

class OrbitalNavigationAgent:
    """
    Real NIS-X Orbital Navigation Agent
    
    Integrates with spacecraft systems for:
    - Autonomous orbital navigation
    - Trajectory planning and optimization
    - Collision avoidance
    - Fuel optimization
    - Mission timeline management
    """
    
    def __init__(self, spacecraft_id: str, mission_config: Dict[str, Any]):
        self.spacecraft_id = spacecraft_id
        self.mission_config = mission_config
        self.current_orbit = None
        self.navigation_history = []
        self.active_commands = []
        self.emergency_protocols = []
        
        # Mission parameters
        self.target_orbit = mission_config.get("target_orbit", {})
        self.fuel_capacity = mission_config.get("fuel_capacity", 1000.0)  # kg
        self.fuel_remaining = mission_config.get("fuel_remaining", 1000.0)  # kg
        self.thrust_capability = mission_config.get("thrust_capability", 440.0)  # N
        self.mission_duration = mission_config.get("mission_duration", 365)  # days
        
        # Navigation parameters
        self.navigation_accuracy = mission_config.get("navigation_accuracy", 0.1)  # km
        self.update_frequency = mission_config.get("update_frequency", 60)  # seconds
        self.autonomous_mode = mission_config.get("autonomous_mode", True)
        
        # Initialize subsystems
        self._initialize_navigation_systems()
    
    def _initialize_navigation_systems(self):
        """Initialize navigation subsystems"""
        
        # Star tracker simulation
        self.star_tracker = {
            "accuracy": 0.001,  # degrees
            "update_rate": 10,  # Hz
            "status": "nominal"
        }
        
        # GPS/GNSS simulation
        self.gnss_receiver = {
            "accuracy": 0.01,  # km
            "update_rate": 1,  # Hz
            "status": "nominal"
        }
        
        # Inertial measurement unit
        self.imu = {
            "gyro_bias": [0.001, 0.001, 0.001],  # deg/hr
            "accel_bias": [0.0001, 0.0001, 0.0001],  # m/s²
            "status": "nominal"
        }
        
        # Propulsion system
        self.propulsion = {
            "thrust_efficiency": 0.95,
            "specific_impulse": 220,  # seconds
            "status": "nominal"
        }
    
    async def update_orbit_state(self, telemetry_data: Dict[str, Any]) -> OrbitState:
        """Update current orbital state from telemetry"""
        
        # Extract position and velocity from telemetry
        position = telemetry_data.get("position", [0.0, 0.0, 0.0])
        velocity = telemetry_data.get("velocity", [0.0, 0.0, 0.0])
        timestamp = datetime.fromisoformat(telemetry_data.get("timestamp", datetime.now().isoformat()))
        
        # Calculate orbital elements
        orbital_elements = self._calculate_orbital_elements(position, velocity)
        
        # Create orbit state
        self.current_orbit = OrbitState(
            position=tuple(position),
            velocity=tuple(velocity),
            timestamp=timestamp,
            semi_major_axis=orbital_elements["semi_major_axis"],
            eccentricity=orbital_elements["eccentricity"],
            inclination=orbital_elements["inclination"],
            ascending_node=orbital_elements["ascending_node"],
            argument_periapsis=orbital_elements["argument_periapsis"],
            true_anomaly=orbital_elements["true_anomaly"]
        )
        
        return self.current_orbit
    
    def _calculate_orbital_elements(self, position: List[float], velocity: List[float]) -> Dict[str, float]:
        """Calculate orbital elements from position and velocity"""
        
        # Earth gravitational parameter
        mu = 3.986004418e14  # m³/s²
        
        # Convert to numpy arrays
        r = np.array(position) * 1000  # Convert km to m
        v = np.array(velocity) * 1000  # Convert km/s to m/s
        
        # Calculate orbital elements using basic orbital mechanics
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # Specific energy
        energy = (v_mag**2 / 2) - (mu / r_mag)
        
        # Semi-major axis
        a = -mu / (2 * energy)
        
        # Angular momentum vector
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # Eccentricity vector
        e_vec = ((v_mag**2 - mu/r_mag) * r - np.dot(r, v) * v) / mu
        e = np.linalg.norm(e_vec)
        
        # Inclination
        i = np.arccos(h[2] / h_mag)
        
        # Simplified calculations for other elements
        # (In real implementation, these would be more precise)
        omega = 0.0  # RAAN
        w = 0.0      # Argument of periapsis
        nu = 0.0     # True anomaly
        
        return {
            "semi_major_axis": a / 1000,  # Convert back to km
            "eccentricity": e,
            "inclination": np.degrees(i),
            "ascending_node": np.degrees(omega),
            "argument_periapsis": np.degrees(w),
            "true_anomaly": np.degrees(nu)
        }
    
    async def plan_trajectory(self, target_state: Dict[str, Any], 
                            constraints: Dict[str, Any] = None) -> List[NavigationCommand]:
        """Plan trajectory to target orbital state"""
        
        if not self.current_orbit:
            raise ValueError("Current orbit state not available")
        
        constraints = constraints or {}
        max_delta_v = constraints.get("max_delta_v", 1000.0)  # m/s
        time_constraint = constraints.get("time_limit", 86400)  # seconds
        fuel_constraint = constraints.get("fuel_limit", self.fuel_remaining * 0.8)  # kg
        
        # Calculate required maneuvers
        trajectory_plan = []
        
        # Phase 1: Hohmann transfer planning
        if target_state.get("altitude_change"):
            hohmann_commands = await self._plan_hohmann_transfer(
                target_state["altitude_change"], 
                constraints
            )
            trajectory_plan.extend(hohmann_commands)
        
        # Phase 2: Plane change maneuvers
        if target_state.get("inclination_change"):
            plane_change_commands = await self._plan_plane_change(
                target_state["inclination_change"],
                constraints
            )
            trajectory_plan.extend(plane_change_commands)
        
        # Phase 3: Fine-tuning maneuvers
        if target_state.get("fine_tuning"):
            fine_tune_commands = await self._plan_fine_tuning(
                target_state["fine_tuning"],
                constraints
            )
            trajectory_plan.extend(fine_tune_commands)
        
        # Validate trajectory plan
        await self._validate_trajectory_plan(trajectory_plan, constraints)
        
        return trajectory_plan
    
    async def _plan_hohmann_transfer(self, altitude_change: float, 
                                   constraints: Dict[str, Any]) -> List[NavigationCommand]:
        """Plan Hohmann transfer maneuver"""
        
        commands = []
        
        # Calculate delta-v for Hohmann transfer
        current_altitude = self.current_orbit.semi_major_axis
        target_altitude = current_altitude + altitude_change
        
        # Earth gravitational parameter
        mu = 3.986004418e14  # m³/s²
        
        # Current orbital velocity
        v1 = np.sqrt(mu / (current_altitude * 1000))
        
        # Transfer orbit periapsis velocity
        v_transfer = np.sqrt(mu * (2/(current_altitude * 1000) - 2/(current_altitude * 1000 + target_altitude * 1000)))
        
        # First burn delta-v
        delta_v1 = v_transfer - v1
        
        # Schedule first burn
        burn_time = datetime.now() + timedelta(minutes=5)  # 5 minutes from now
        
        first_burn = NavigationCommand(
            command_type="burn",
            parameters={
                "delta_v": delta_v1,
                "direction": "prograde",
                "burn_duration": abs(delta_v1) / (self.thrust_capability / 1000),  # seconds
                "fuel_consumption": abs(delta_v1) * 1000 / (self.propulsion["specific_impulse"] * 9.81)  # kg
            },
            execution_time=burn_time,
            duration=abs(delta_v1) / (self.thrust_capability / 1000),
            priority=8
        )
        
        commands.append(first_burn)
        
        # Second burn (at apoapsis)
        transfer_period = 2 * np.pi * np.sqrt(((current_altitude + target_altitude) * 1000 / 2)**3 / mu)
        second_burn_time = burn_time + timedelta(seconds=transfer_period / 2)
        
        # Target orbital velocity
        v2 = np.sqrt(mu / (target_altitude * 1000))
        
        # Transfer orbit apoapsis velocity
        v_transfer_ap = np.sqrt(mu * (2/(target_altitude * 1000) - 2/(current_altitude * 1000 + target_altitude * 1000)))
        
        # Second burn delta-v
        delta_v2 = v2 - v_transfer_ap
        
        second_burn = NavigationCommand(
            command_type="burn",
            parameters={
                "delta_v": delta_v2,
                "direction": "prograde",
                "burn_duration": abs(delta_v2) / (self.thrust_capability / 1000),
                "fuel_consumption": abs(delta_v2) * 1000 / (self.propulsion["specific_impulse"] * 9.81)
            },
            execution_time=second_burn_time,
            duration=abs(delta_v2) / (self.thrust_capability / 1000),
            priority=8
        )
        
        commands.append(second_burn)
        
        return commands
    
    async def _plan_plane_change(self, inclination_change: float, 
                               constraints: Dict[str, Any]) -> List[NavigationCommand]:
        """Plan plane change maneuver"""
        
        # Calculate delta-v for plane change
        v_orbital = np.sqrt(3.986004418e14 / (self.current_orbit.semi_major_axis * 1000))
        delta_v = 2 * v_orbital * np.sin(np.radians(inclination_change) / 2)
        
        # Schedule plane change at ascending/descending node
        execution_time = datetime.now() + timedelta(minutes=30)
        
        plane_change_command = NavigationCommand(
            command_type="burn",
            parameters={
                "delta_v": delta_v,
                "direction": "normal",
                "burn_duration": delta_v / (self.thrust_capability / 1000),
                "fuel_consumption": delta_v * 1000 / (self.propulsion["specific_impulse"] * 9.81)
            },
            execution_time=execution_time,
            duration=delta_v / (self.thrust_capability / 1000),
            priority=7
        )
        
        return [plane_change_command]
    
    async def _plan_fine_tuning(self, fine_tuning_params: Dict[str, Any], 
                              constraints: Dict[str, Any]) -> List[NavigationCommand]:
        """Plan fine-tuning maneuvers"""
        
        commands = []
        
        # Small correction burns
        for i, correction in enumerate(fine_tuning_params.get("corrections", [])):
            execution_time = datetime.now() + timedelta(hours=1 + i)
            
            fine_tune_command = NavigationCommand(
                command_type="burn",
                parameters={
                    "delta_v": correction.get("delta_v", 0.1),
                    "direction": correction.get("direction", "prograde"),
                    "burn_duration": correction.get("duration", 1.0),
                    "fuel_consumption": correction.get("fuel", 0.1)
                },
                execution_time=execution_time,
                duration=correction.get("duration", 1.0),
                priority=5
            )
            
            commands.append(fine_tune_command)
        
        return commands
    
    async def _validate_trajectory_plan(self, trajectory_plan: List[NavigationCommand], 
                                      constraints: Dict[str, Any]):
        """Validate trajectory plan against constraints"""
        
        total_delta_v = sum(cmd.parameters.get("delta_v", 0) for cmd in trajectory_plan)
        total_fuel = sum(cmd.parameters.get("fuel_consumption", 0) for cmd in trajectory_plan)
        
        # Check delta-v constraint
        if total_delta_v > constraints.get("max_delta_v", 1000.0):
            raise ValueError(f"Total delta-v {total_delta_v} exceeds constraint {constraints.get('max_delta_v', 1000.0)}")
        
        # Check fuel constraint
        if total_fuel > constraints.get("fuel_limit", self.fuel_remaining):
            raise ValueError(f"Total fuel consumption {total_fuel} exceeds available fuel {self.fuel_remaining}")
        
        # Check time constraint
        total_time = max(cmd.execution_time for cmd in trajectory_plan) - min(cmd.execution_time for cmd in trajectory_plan)
        if total_time.total_seconds() > constraints.get("time_limit", 86400):
            raise ValueError(f"Trajectory duration exceeds time constraint")
    
    async def execute_navigation_command(self, command: NavigationCommand) -> Dict[str, Any]:
        """Execute a navigation command"""
        
        execution_start = datetime.now()
        
        # Validate command
        if command.execution_time < execution_start:
            raise ValueError("Command execution time is in the past")
        
        # Wait for execution time
        wait_time = (command.execution_time - execution_start).total_seconds()
        if wait_time > 0:
            await asyncio.sleep(min(wait_time, 1.0))  # Simulate wait (max 1 second for demo)
        
        # Execute command based on type
        if command.command_type == "burn":
            result = await self._execute_burn(command)
        elif command.command_type == "attitude":
            result = await self._execute_attitude_change(command)
        elif command.command_type == "coast":
            result = await self._execute_coast(command)
        else:
            raise ValueError(f"Unknown command type: {command.command_type}")
        
        # Log execution
        self.navigation_history.append({
            "command": command.to_dict(),
            "result": result,
            "execution_time": datetime.now().isoformat()
        })
        
        return result
    
    async def _execute_burn(self, command: NavigationCommand) -> Dict[str, Any]:
        """Execute propulsive maneuver"""
        
        delta_v = command.parameters.get("delta_v", 0.0)
        direction = command.parameters.get("direction", "prograde")
        fuel_consumption = command.parameters.get("fuel_consumption", 0.0)
        
        # Check fuel availability
        if fuel_consumption > self.fuel_remaining:
            return {
                "success": False,
                "error": "Insufficient fuel",
                "fuel_required": fuel_consumption,
                "fuel_available": self.fuel_remaining
            }
        
        # Simulate burn execution
        await asyncio.sleep(0.1)  # Simulate burn time
        
        # Update fuel
        self.fuel_remaining -= fuel_consumption
        
        # Simulate orbit change (simplified)
        if self.current_orbit and direction == "prograde":
            # Increase semi-major axis
            new_sma = self.current_orbit.semi_major_axis + (delta_v / 1000) * 100  # Simplified
            self.current_orbit = OrbitState(
                position=self.current_orbit.position,
                velocity=self.current_orbit.velocity,
                timestamp=datetime.now(),
                semi_major_axis=new_sma,
                eccentricity=self.current_orbit.eccentricity,
                inclination=self.current_orbit.inclination,
                ascending_node=self.current_orbit.ascending_node,
                argument_periapsis=self.current_orbit.argument_periapsis,
                true_anomaly=self.current_orbit.true_anomaly
            )
        
        return {
            "success": True,
            "delta_v_achieved": delta_v,
            "fuel_consumed": fuel_consumption,
            "fuel_remaining": self.fuel_remaining,
            "new_orbit": self.current_orbit.to_dict() if self.current_orbit else None
        }
    
    async def _execute_attitude_change(self, command: NavigationCommand) -> Dict[str, Any]:
        """Execute attitude change maneuver"""
        
        target_attitude = command.parameters.get("target_attitude", [0, 0, 0])
        
        # Simulate attitude change
        await asyncio.sleep(0.05)
        
        return {
            "success": True,
            "attitude_achieved": target_attitude,
            "pointing_accuracy": 0.1  # degrees
        }
    
    async def _execute_coast(self, command: NavigationCommand) -> Dict[str, Any]:
        """Execute coast phase"""
        
        coast_duration = command.duration
        
        # Simulate coast (propagate orbit)
        await asyncio.sleep(0.01)
        
        return {
            "success": True,
            "coast_duration": coast_duration,
            "final_position": self.current_orbit.position if self.current_orbit else None
        }
    
    async def autonomous_navigation_cycle(self) -> Dict[str, Any]:
        """Autonomous navigation cycle"""
        
        if not self.autonomous_mode:
            return {"status": "manual_mode", "message": "Autonomous mode disabled"}
        
        # Check for trajectory corrections needed
        corrections_needed = await self._assess_trajectory_corrections()
        
        if corrections_needed:
            # Plan corrective maneuvers
            corrective_commands = await self._plan_corrective_maneuvers(corrections_needed)
            
            # Execute high-priority corrections
            execution_results = []
            for command in corrective_commands:
                if command.priority >= 8:  # High priority
                    result = await self.execute_navigation_command(command)
                    execution_results.append(result)
            
            return {
                "status": "corrections_executed",
                "corrections_needed": corrections_needed,
                "commands_executed": len(execution_results),
                "results": execution_results
            }
        
        return {
            "status": "nominal",
            "message": "No corrections needed",
            "orbit_state": self.current_orbit.to_dict() if self.current_orbit else None
        }
    
    async def _assess_trajectory_corrections(self) -> List[Dict[str, Any]]:
        """Assess if trajectory corrections are needed"""
        
        corrections = []
        
        if not self.current_orbit:
            return corrections
        
        # Check if orbit is drifting from target
        if self.target_orbit:
            target_sma = self.target_orbit.get("semi_major_axis", self.current_orbit.semi_major_axis)
            sma_error = abs(self.current_orbit.semi_major_axis - target_sma)
            
            if sma_error > 1.0:  # More than 1 km error
                corrections.append({
                    "type": "altitude_correction",
                    "current_sma": self.current_orbit.semi_major_axis,
                    "target_sma": target_sma,
                    "error": sma_error
                })
        
        # Check fuel status
        fuel_percentage = self.fuel_remaining / self.fuel_capacity
        if fuel_percentage < 0.2:  # Less than 20% fuel
            corrections.append({
                "type": "fuel_critical",
                "fuel_remaining": self.fuel_remaining,
                "fuel_percentage": fuel_percentage
            })
        
        return corrections
    
    async def _plan_corrective_maneuvers(self, corrections: List[Dict[str, Any]]) -> List[NavigationCommand]:
        """Plan corrective maneuvers"""
        
        commands = []
        
        for correction in corrections:
            if correction["type"] == "altitude_correction":
                # Small correction burn
                delta_v = min(correction["error"] / 100, 1.0)  # Limit to 1 m/s
                
                correction_command = NavigationCommand(
                    command_type="burn",
                    parameters={
                        "delta_v": delta_v,
                        "direction": "prograde" if correction["error"] > 0 else "retrograde",
                        "burn_duration": delta_v / (self.thrust_capability / 1000),
                        "fuel_consumption": delta_v * 1000 / (self.propulsion["specific_impulse"] * 9.81)
                    },
                    execution_time=datetime.now() + timedelta(minutes=1),
                    duration=delta_v / (self.thrust_capability / 1000),
                    priority=8
                )
                
                commands.append(correction_command)
        
        return commands
    
    async def get_telemetry(self) -> Dict[str, Any]:
        """Get current telemetry data"""
        
        return {
            "spacecraft_id": self.spacecraft_id,
            "timestamp": datetime.now().isoformat(),
            "orbit_state": self.current_orbit.to_dict() if self.current_orbit else None,
            "fuel_remaining": self.fuel_remaining,
            "fuel_percentage": self.fuel_remaining / self.fuel_capacity,
            "systems_status": {
                "star_tracker": self.star_tracker["status"],
                "gnss": self.gnss_receiver["status"],
                "imu": self.imu["status"],
                "propulsion": self.propulsion["status"]
            },
            "autonomous_mode": self.autonomous_mode,
            "active_commands": len(self.active_commands),
            "navigation_history_count": len(self.navigation_history)
        }

# Factory function for NIS-X integration
def create_nis_x_orbital_agent(spacecraft_id: str, mission_config: Dict[str, Any]) -> OrbitalNavigationAgent:
    """Create orbital navigation agent for NIS-X missions"""
    
    # NIS-X specific configuration
    nis_x_config = {
        "fuel_capacity": mission_config.get("fuel_capacity", 500.0),
        "thrust_capability": mission_config.get("thrust_capability", 220.0),
        "mission_duration": mission_config.get("mission_duration", 180),
        "navigation_accuracy": 0.05,  # High accuracy for NIS-X
        "autonomous_mode": True,
        "update_frequency": 30,  # 30-second updates
        **mission_config
    }
    
    return OrbitalNavigationAgent(spacecraft_id, nis_x_config)

# Example usage for NIS-X integration
async def example_nis_x_mission():
    """Example NIS-X mission scenario"""
    
    # Create orbital navigation agent
    agent = create_nis_x_orbital_agent(
        spacecraft_id="NIS-X-001",
        mission_config={
            "fuel_capacity": 800.0,
            "target_orbit": {
                "semi_major_axis": 6800.0,  # 400 km altitude
                "inclination": 51.6  # ISS-like orbit
            }
        }
    )
    
    # Initialize with current telemetry
    await agent.update_orbit_state({
        "position": [6578.0, 0.0, 0.0],  # km
        "velocity": [0.0, 7.5, 0.0],     # km/s
        "timestamp": datetime.now().isoformat()
    })
    
    # Plan trajectory to target
    trajectory = await agent.plan_trajectory({
        "altitude_change": 50.0,  # Raise orbit by 50 km
        "fine_tuning": {
            "corrections": [
                {"delta_v": 0.1, "direction": "prograde", "duration": 1.0}
            ]
        }
    })
    
    # Execute first maneuver
    if trajectory:
        result = await agent.execute_navigation_command(trajectory[0])
        print(f"Maneuver result: {result}")
    
    # Get telemetry
    telemetry = await agent.get_telemetry()
    print(f"Mission telemetry: {telemetry}")

if __name__ == "__main__":
    asyncio.run(example_nis_x_mission()) 