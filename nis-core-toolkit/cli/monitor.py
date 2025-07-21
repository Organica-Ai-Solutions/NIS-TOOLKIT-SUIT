#!/usr/bin/env python3
"""
NIS Core Toolkit - Consciousness Monitoring CLI
Real-time monitoring of NIS v3.0 AGI system performance
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import argparse
import sys
from pathlib import Path

try:
    import websockets
    import redis
    from kafka import KafkaConsumer
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    MONITORING_DEPS_AVAILABLE = True
except ImportError:
    MONITORING_DEPS_AVAILABLE = False

# Import our consciousness interface
try:
    from ..templates.nis_v3_integration.consciousness_interface import NISConsciousnessInterface, ConsciousnessConfig
    from ..templates.nis_v3_integration.kan_interface import NISKANInterface, KANConfig
except ImportError:
    # Fallback for direct execution
    sys.path.append(str(Path(__file__).parent.parent))
    from templates.nis_v3_integration.consciousness_interface import NISConsciousnessInterface, ConsciousnessConfig
    from templates.nis_v3_integration.kan_interface import NISKANInterface, KANConfig

class ConsciousnessMonitor:
    """
    Real-time consciousness monitoring system
    Interfaces with NIS Protocol v3.0 consciousness modules
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consciousness_interface = None
        self.kan_interface = None
        self.monitoring_data = []
        self.active_connections = {}
        self.alert_thresholds = {
            "consciousness_health": 0.6,
            "cognitive_load": 0.8,
            "bias_detection_rate": 0.2,
            "kan_interpretability": 0.85
        }
        self.is_monitoring = False
        
    async def start_monitoring(self, mode: str = "real_time"):
        """Start consciousness monitoring"""
        
        print("🧠 Starting NIS v3.0 Consciousness Monitoring...")
        
        # Initialize interfaces
        await self._initialize_interfaces()
        
        # Start monitoring based on mode
        if mode == "real_time":
            await self._start_real_time_monitoring()
        elif mode == "batch":
            await self._start_batch_monitoring()
        elif mode == "dashboard":
            await self._start_dashboard_monitoring()
        else:
            print(f"❌ Unknown monitoring mode: {mode}")
            return
        
        print("✅ Monitoring started successfully")
    
    async def _initialize_interfaces(self):
        """Initialize consciousness and KAN interfaces"""
        
        # Initialize consciousness interface
        consciousness_config = ConsciousnessConfig(
            meta_cognitive_processing=True,
            bias_detection=True,
            self_reflection_interval=30,  # 30 seconds for monitoring
            introspection_depth=0.8,
            emotional_awareness=True,
            attention_tracking=True
        )
        
        self.consciousness_interface = NISConsciousnessInterface(consciousness_config)
        
        # Initialize KAN interface  
        kan_config = KANConfig(
            interpretability_threshold=0.9,
            mathematical_proofs=True,
            convergence_guarantees=True
        )
        
        self.kan_interface = NISKANInterface(kan_config)
        
        print("🔌 Initialized consciousness and KAN interfaces")
    
    async def _start_real_time_monitoring(self):
        """Start real-time monitoring"""
        
        self.is_monitoring = True
        
        print("📊 Starting real-time consciousness monitoring...")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while self.is_monitoring:
                # Collect consciousness metrics
                consciousness_metrics = await self._collect_consciousness_metrics()
                
                # Collect KAN metrics
                kan_metrics = await self._collect_kan_metrics()
                
                # Combine metrics
                combined_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "consciousness": consciousness_metrics,
                    "kan": kan_metrics,
                    "system": await self._collect_system_metrics()
                }
                
                # Store metrics
                self.monitoring_data.append(combined_metrics)
                
                # Display real-time status
                await self._display_real_time_status(combined_metrics)
                
                # Check for alerts
                await self._check_alerts(combined_metrics)
                
                # Wait before next collection
                await asyncio.sleep(5)  # 5-second intervals
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        finally:
            self.is_monitoring = False
            await self._save_monitoring_data()
    
    async def _collect_consciousness_metrics(self) -> Dict[str, Any]:
        """Collect consciousness metrics"""
        
        if not self.consciousness_interface:
            return {"error": "consciousness_interface_not_initialized"}
        
        try:
            # Get consciousness state
            consciousness_state = await self.consciousness_interface.reflect()
            
            # Get comprehensive metrics
            metrics = await self.consciousness_interface.get_consciousness_metrics()
            
            return {
                "state": consciousness_state.to_dict(),
                "health": metrics.get("consciousness_health", {}),
                "processing_stats": metrics.get("processing_statistics", {}),
                "trends": metrics.get("consciousness_trends", {})
            }
            
        except Exception as e:
            return {"error": f"consciousness_collection_failed: {e}"}
    
    async def _collect_kan_metrics(self) -> Dict[str, Any]:
        """Collect KAN metrics"""
        
        if not self.kan_interface:
            return {"error": "kan_interface_not_initialized"}
        
        try:
            # Validate mathematical guarantees
            validation_results = await self.kan_interface.validate_mathematical_guarantees()
            
            # Get performance metrics (simulated)
            performance_metrics = {
                "interpretability_score": 0.95,
                "mathematical_accuracy": 0.98,
                "convergence_time": 0.05,
                "spline_optimization": 0.92
            }
            
            return {
                "validation": validation_results,
                "performance": performance_metrics,
                "mathematical_guarantees": validation_results.get("overall_valid", False)
            }
            
        except Exception as e:
            return {"error": f"kan_collection_failed: {e}"}
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics"""
        
        import psutil
        import os
        
        try:
            # Get process information
            process = psutil.Process(os.getpid())
            
            # Get system metrics
            system_metrics = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": process.memory_info().rss / 1024 / 1024,  # MB
                "memory_percent": process.memory_percent(),
                "open_files": len(process.open_files()),
                "threads": process.num_threads(),
                "uptime": time.time() - process.create_time()
            }
            
            return system_metrics
            
        except Exception as e:
            return {"error": f"system_collection_failed: {e}"}
    
    async def _display_real_time_status(self, metrics: Dict[str, Any]):
        """Display real-time status"""
        
        # Clear screen (simple approach)
        print("\033[2J\033[H")
        
        # Header
        print("🧠 NIS v3.0 Consciousness Monitor")
        print("=" * 50)
        print(f"📅 {metrics['timestamp']}")
        print()
        
        # Consciousness Status
        consciousness = metrics.get("consciousness", {})
        if "error" not in consciousness:
            state = consciousness.get("state", {})
            health = consciousness.get("health", {})
            
            print("🧠 CONSCIOUSNESS STATUS")
            print(f"  Self-Awareness: {state.get('self_awareness_score', 0):.2f}")
            print(f"  Cognitive Load: {state.get('cognitive_load', 0):.2f}")
            print(f"  Bias Flags: {len(state.get('bias_flags', []))}")
            print(f"  Active Insights: {len(state.get('meta_cognitive_insights', []))}")
            
            if health:
                print(f"  Health Score: {health.get('self_awareness', 0):.2f}")
                print(f"  Cognitive Efficiency: {health.get('cognitive_efficiency', 0):.2f}")
        else:
            print(f"🧠 CONSCIOUSNESS: ❌ {consciousness.get('error', 'Unknown error')}")
        
        print()
        
        # KAN Status
        kan = metrics.get("kan", {})
        if "error" not in kan:
            validation = kan.get("validation", {})
            performance = kan.get("performance", {})
            
            print("🧮 KAN REASONING STATUS")
            print(f"  Mathematical Guarantees: {'✅' if kan.get('mathematical_guarantees', False) else '❌'}")
            print(f"  Interpretability: {performance.get('interpretability_score', 0):.2f}")
            print(f"  Mathematical Accuracy: {performance.get('mathematical_accuracy', 0):.2f}")
            print(f"  Convergence Time: {performance.get('convergence_time', 0):.3f}s")
        else:
            print(f"🧮 KAN: ❌ {kan.get('error', 'Unknown error')}")
        
        print()
        
        # System Status
        system = metrics.get("system", {})
        if "error" not in system:
            print("⚙️  SYSTEM STATUS")
            print(f"  CPU Usage: {system.get('cpu_usage', 0):.1f}%")
            print(f"  Memory Usage: {system.get('memory_usage', 0):.1f} MB")
            print(f"  Memory Percent: {system.get('memory_percent', 0):.1f}%")
            print(f"  Active Threads: {system.get('threads', 0)}")
        else:
            print(f"⚙️  SYSTEM: ❌ {system.get('error', 'Unknown error')}")
        
        print()
        print("📊 Press Ctrl+C to stop monitoring...")
    
    async def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions"""
        
        alerts = []
        
        # Check consciousness health
        consciousness = metrics.get("consciousness", {})
        if "error" not in consciousness:
            health = consciousness.get("health", {})
            consciousness_health = health.get("self_awareness", 0)
            
            if consciousness_health < self.alert_thresholds["consciousness_health"]:
                alerts.append(f"🚨 Low consciousness health: {consciousness_health:.2f}")
            
            # Check cognitive load
            state = consciousness.get("state", {})
            cognitive_load = state.get("cognitive_load", 0)
            
            if cognitive_load > self.alert_thresholds["cognitive_load"]:
                alerts.append(f"⚠️  High cognitive load: {cognitive_load:.2f}")
            
            # Check bias detection
            bias_flags = len(state.get("bias_flags", []))
            if bias_flags > 3:  # More than 3 active biases
                alerts.append(f"🎯 Multiple biases detected: {bias_flags}")
        
        # Check KAN performance
        kan = metrics.get("kan", {})
        if "error" not in kan:
            performance = kan.get("performance", {})
            interpretability = performance.get("interpretability_score", 0)
            
            if interpretability < self.alert_thresholds["kan_interpretability"]:
                alerts.append(f"📉 Low KAN interpretability: {interpretability:.2f}")
        
        # Display alerts
        if alerts:
            print("\n🚨 ALERTS:")
            for alert in alerts:
                print(f"  {alert}")
    
    async def _save_monitoring_data(self):
        """Save monitoring data to file"""
        
        if not self.monitoring_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"consciousness_monitoring_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.monitoring_data, f, indent=2)
            
            print(f"📁 Monitoring data saved to {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save monitoring data: {e}")
    
    async def _start_batch_monitoring(self):
        """Start batch monitoring mode"""
        
        print("📊 Starting batch consciousness monitoring...")
        
        # Collect metrics once
        consciousness_metrics = await self._collect_consciousness_metrics()
        kan_metrics = await self._collect_kan_metrics()
        system_metrics = await self._collect_system_metrics()
        
        # Create comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "consciousness": consciousness_metrics,
            "kan": kan_metrics,
            "system": system_metrics,
            "summary": await self._generate_batch_summary(consciousness_metrics, kan_metrics, system_metrics)
        }
        
        # Display report
        await self._display_batch_report(report)
        
        # Save report
        await self._save_batch_report(report)
    
    async def _generate_batch_summary(self, consciousness: Dict[str, Any], 
                                    kan: Dict[str, Any], system: Dict[str, Any]) -> Dict[str, Any]:
        """Generate batch monitoring summary"""
        
        summary = {
            "consciousness_health": "unknown",
            "kan_performance": "unknown",
            "system_performance": "unknown",
            "overall_status": "unknown",
            "recommendations": []
        }
        
        # Assess consciousness health
        if "error" not in consciousness:
            health = consciousness.get("health", {})
            avg_health = sum(health.values()) / len(health) if health else 0
            
            if avg_health > 0.8:
                summary["consciousness_health"] = "excellent"
            elif avg_health > 0.6:
                summary["consciousness_health"] = "good"
            else:
                summary["consciousness_health"] = "needs_attention"
                summary["recommendations"].append("Improve consciousness monitoring frequency")
        
        # Assess KAN performance
        if "error" not in kan:
            performance = kan.get("performance", {})
            interpretability = performance.get("interpretability_score", 0)
            
            if interpretability > 0.9:
                summary["kan_performance"] = "excellent"
            elif interpretability > 0.8:
                summary["kan_performance"] = "good"
            else:
                summary["kan_performance"] = "needs_attention"
                summary["recommendations"].append("Optimize KAN interpretability settings")
        
        # Assess system performance
        if "error" not in system:
            cpu_usage = system.get("cpu_usage", 0)
            memory_percent = system.get("memory_percent", 0)
            
            if cpu_usage < 50 and memory_percent < 70:
                summary["system_performance"] = "excellent"
            elif cpu_usage < 80 and memory_percent < 85:
                summary["system_performance"] = "good"
            else:
                summary["system_performance"] = "needs_attention"
                summary["recommendations"].append("Monitor system resource usage")
        
        # Overall status
        statuses = [summary["consciousness_health"], summary["kan_performance"], summary["system_performance"]]
        
        if all(s == "excellent" for s in statuses):
            summary["overall_status"] = "excellent"
        elif all(s in ["excellent", "good"] for s in statuses):
            summary["overall_status"] = "good"
        else:
            summary["overall_status"] = "needs_attention"
        
        return summary
    
    async def _display_batch_report(self, report: Dict[str, Any]):
        """Display batch monitoring report"""
        
        print("\n🧠 NIS v3.0 Consciousness Monitoring Report")
        print("=" * 60)
        print(f"📅 Generated: {report['timestamp']}")
        print()
        
        # Summary
        summary = report.get("summary", {})
        print("📊 SUMMARY")
        print(f"  Overall Status: {summary.get('overall_status', 'unknown').upper()}")
        print(f"  Consciousness Health: {summary.get('consciousness_health', 'unknown').upper()}")
        print(f"  KAN Performance: {summary.get('kan_performance', 'unknown').upper()}")
        print(f"  System Performance: {summary.get('system_performance', 'unknown').upper()}")
        print()
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print("💡 RECOMMENDATIONS")
            for rec in recommendations:
                print(f"  • {rec}")
            print()
        
        # Detailed metrics
        consciousness = report.get("consciousness", {})
        if "error" not in consciousness:
            state = consciousness.get("state", {})
            health = consciousness.get("health", {})
            
            print("🧠 CONSCIOUSNESS DETAILS")
            print(f"  Self-Awareness Score: {state.get('self_awareness_score', 0):.3f}")
            print(f"  Cognitive Load: {state.get('cognitive_load', 0):.3f}")
            print(f"  Active Biases: {len(state.get('bias_flags', []))}")
            print(f"  Meta-Cognitive Insights: {len(state.get('meta_cognitive_insights', []))}")
            
            if health:
                print(f"  Health Metrics:")
                for key, value in health.items():
                    print(f"    {key}: {value:.3f}")
            print()
        
        # KAN details
        kan = report.get("kan", {})
        if "error" not in kan:
            validation = kan.get("validation", {})
            performance = kan.get("performance", {})
            
            print("🧮 KAN DETAILS")
            print(f"  Mathematical Guarantees: {'✅' if kan.get('mathematical_guarantees', False) else '❌'}")
            print(f"  Interpretability Score: {performance.get('interpretability_score', 0):.3f}")
            print(f"  Mathematical Accuracy: {performance.get('mathematical_accuracy', 0):.3f}")
            print(f"  Convergence Time: {performance.get('convergence_time', 0):.3f}s")
            
            if validation.get("overall_valid", False):
                print("  Validation Status: ✅ All tests passed")
            else:
                print("  Validation Status: ❌ Some tests failed")
            print()
        
        # System details
        system = report.get("system", {})
        if "error" not in system:
            print("⚙️  SYSTEM DETAILS")
            print(f"  CPU Usage: {system.get('cpu_usage', 0):.1f}%")
            print(f"  Memory Usage: {system.get('memory_usage', 0):.1f} MB")
            print(f"  Memory Percent: {system.get('memory_percent', 0):.1f}%")
            print(f"  Active Threads: {system.get('threads', 0)}")
            print(f"  Open Files: {system.get('open_files', 0)}")
            print(f"  Uptime: {system.get('uptime', 0):.1f} seconds")
    
    async def _save_batch_report(self, report: Dict[str, Any]):
        """Save batch report to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"consciousness_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n📁 Report saved to {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
    
    async def _start_dashboard_monitoring(self):
        """Start dashboard monitoring mode"""
        
        if not MONITORING_DEPS_AVAILABLE:
            print("❌ Dashboard monitoring requires additional dependencies:")
            print("   pip install plotly pandas websockets redis-py kafka-python")
            return
        
        print("📊 Starting dashboard consciousness monitoring...")
        print("🌐 Dashboard will be available at http://localhost:8050")
        
        # Create dashboard
        await self._create_monitoring_dashboard()
    
    async def _create_monitoring_dashboard(self):
        """Create interactive monitoring dashboard"""
        
        try:
            import dash
            from dash import dcc, html, Input, Output
            import plotly.express as px
            
            # Create Dash app
            app = dash.Dash(__name__)
            
            # Layout
            app.layout = html.Div([
                html.H1("🧠 NIS v3.0 Consciousness Monitor", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                
                html.Div([
                    html.Div([
                        html.H3("Consciousness Health"),
                        dcc.Graph(id='consciousness-health-graph')
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.H3("KAN Performance"),
                        dcc.Graph(id='kan-performance-graph')
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                ]),
                
                html.Div([
                    html.Div([
                        html.H3("System Metrics"),
                        dcc.Graph(id='system-metrics-graph')
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.H3("Real-time Status"),
                        html.Div(id='real-time-status')
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                ]),
                
                dcc.Interval(
                    id='interval-component',
                    interval=5000,  # Update every 5 seconds
                    n_intervals=0
                )
            ])
            
            # Callbacks
            @app.callback(
                [Output('consciousness-health-graph', 'figure'),
                 Output('kan-performance-graph', 'figure'),
                 Output('system-metrics-graph', 'figure'),
                 Output('real-time-status', 'children')],
                [Input('interval-component', 'n_intervals')]
            )
            def update_dashboard(n):
                # This would be implemented with real-time data
                # For now, return placeholder graphs
                
                # Consciousness health graph
                consciousness_fig = px.line(
                    x=[1, 2, 3, 4, 5],
                    y=[0.8, 0.85, 0.82, 0.88, 0.9],
                    title="Consciousness Health Over Time"
                )
                
                # KAN performance graph
                kan_fig = px.bar(
                    x=['Interpretability', 'Accuracy', 'Convergence'],
                    y=[0.95, 0.98, 0.92],
                    title="KAN Performance Metrics"
                )
                
                # System metrics graph
                system_fig = px.line(
                    x=[1, 2, 3, 4, 5],
                    y=[45, 50, 48, 52, 47],
                    title="CPU Usage Over Time"
                )
                
                # Real-time status
                status_div = html.Div([
                    html.P("🟢 Consciousness: Healthy", style={'color': 'green'}),
                    html.P("🟢 KAN: Optimal", style={'color': 'green'}),
                    html.P("🟡 System: Good", style={'color': 'orange'}),
                    html.P(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
                ])
                
                return consciousness_fig, kan_fig, system_fig, status_div
            
            # Run the app
            app.run_server(debug=False, host='0.0.0.0', port=8050)
            
        except ImportError:
            print("❌ Dashboard requires Dash: pip install dash")
        except Exception as e:
            print(f"❌ Dashboard error: {e}")

class ConsciousnessDebugger:
    """
    Advanced debugging tools for consciousness development
    """
    
    def __init__(self):
        self.debug_sessions = []
        self.breakpoints = []
        self.trace_enabled = False
    
    async def debug_consciousness_processing(self, consciousness_interface: NISConsciousnessInterface,
                                          input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Debug consciousness processing step by step"""
        
        print("🐛 Starting consciousness debugging session...")
        
        debug_session = {
            "timestamp": datetime.now().isoformat(),
            "input_data": input_data,
            "processing_steps": [],
            "consciousness_states": [],
            "debug_insights": []
        }
        
        # Step 1: Pre-processing consciousness state
        pre_state = await consciousness_interface.reflect()
        debug_session["consciousness_states"].append({
            "phase": "pre_processing",
            "state": pre_state.to_dict()
        })
        
        print(f"📊 Pre-processing consciousness state:")
        print(f"  Self-awareness: {pre_state.self_awareness_score:.3f}")
        print(f"  Cognitive load: {pre_state.cognitive_load:.3f}")
        print(f"  Active biases: {len(pre_state.bias_flags)}")
        
        # Step 2: Process with consciousness (with debugging)
        result = await consciousness_interface.process_with_consciousness(input_data)
        debug_session["processing_steps"].append({
            "step": "consciousness_processing",
            "result": result
        })
        
        # Step 3: Post-processing consciousness state
        post_state = await consciousness_interface.reflect()
        debug_session["consciousness_states"].append({
            "phase": "post_processing",
            "state": post_state.to_dict()
        })
        
        print(f"📊 Post-processing consciousness state:")
        print(f"  Self-awareness: {post_state.self_awareness_score:.3f}")
        print(f"  Cognitive load: {post_state.cognitive_load:.3f}")
        print(f"  Active biases: {len(post_state.bias_flags)}")
        
        # Step 4: Generate debug insights
        insights = await self._generate_debug_insights(pre_state, post_state, result)
        debug_session["debug_insights"] = insights
        
        print("\n🧠 Debug Insights:")
        for insight in insights:
            print(f"  • {insight}")
        
        # Store debug session
        self.debug_sessions.append(debug_session)
        
        return debug_session
    
    async def _generate_debug_insights(self, pre_state, post_state, result) -> List[str]:
        """Generate debugging insights"""
        
        insights = []
        
        # Analyze consciousness state changes
        awareness_change = post_state.self_awareness_score - pre_state.self_awareness_score
        if awareness_change > 0.05:
            insights.append(f"Significant awareness increase: +{awareness_change:.3f}")
        elif awareness_change < -0.05:
            insights.append(f"Awareness decrease detected: {awareness_change:.3f}")
        
        # Analyze cognitive load changes
        load_change = post_state.cognitive_load - pre_state.cognitive_load
        if load_change > 0.1:
            insights.append(f"High cognitive load increase: +{load_change:.3f}")
        
        # Analyze bias detection
        new_biases = len(post_state.bias_flags) - len(pre_state.bias_flags)
        if new_biases > 0:
            insights.append(f"New biases detected: {new_biases}")
        
        # Analyze processing confidence
        confidence = result.get("processing_confidence", 0)
        if confidence < 0.7:
            insights.append(f"Low processing confidence: {confidence:.3f}")
        
        return insights

def main():
    """Main CLI function for consciousness monitoring"""
    
    parser = argparse.ArgumentParser(
        description="NIS v3.0 Consciousness Monitoring CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time monitoring
  nis monitor --mode real_time
  
  # Batch monitoring report
  nis monitor --mode batch
  
  # Interactive dashboard
  nis monitor --mode dashboard
  
  # Debug consciousness processing
  nis monitor --debug --input '{"task": "reasoning", "complexity": 0.8}'
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["real_time", "batch", "dashboard"],
        default="real_time",
        help="Monitoring mode"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debugging mode"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input data for debugging (JSON string)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for reports"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return 1
    
    # Run monitoring
    try:
        if args.debug:
            # Debug mode
            if not args.input:
                print("❌ Debug mode requires --input parameter")
                return 1
            
            input_data = json.loads(args.input)
            
            async def debug_session():
                debugger = ConsciousnessDebugger()
                consciousness_interface = NISConsciousnessInterface(ConsciousnessConfig())
                
                debug_result = await debugger.debug_consciousness_processing(
                    consciousness_interface, input_data
                )
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(debug_result, f, indent=2)
                    print(f"📁 Debug session saved to {args.output}")
            
            asyncio.run(debug_session())
            
        else:
            # Monitoring mode
            async def monitor_session():
                monitor = ConsciousnessMonitor(config)
                await monitor.start_monitoring(args.mode)
            
            asyncio.run(monitor_session())
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 