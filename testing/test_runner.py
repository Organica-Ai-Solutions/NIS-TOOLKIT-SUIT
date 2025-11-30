#!/usr/bin/env python3
"""
NIS TOOLKIT SUIT v4.0.0 - Comprehensive Test Runner
Advanced testing framework with coverage, benchmarking, security scanning, and quality analysis
"""

import os
import sys
import json
import time
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yaml
    import pytest
    from coverage import Coverage
except ImportError:
    print("Installing required testing packages...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 
                   'pyyaml', 'pytest', 'coverage', 'pytest-cov', 'pytest-benchmark'])

@dataclass
class TestResult:
    """Test result container"""
    name: str
    status: str  # passed, failed, skipped, error
    duration: float
    message: str = ""
    traceback: str = ""
    
@dataclass  
class TestSuite:
    """Test suite results"""
    name: str
    tests: List[TestResult]
    total_duration: float
    passed: int
    failed: int
    skipped: int
    errors: int

@dataclass
class TestReport:
    """Comprehensive test report"""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_duration: float
    coverage_percentage: float
    security_issues: int
    quality_score: float
    performance_metrics: Dict[str, Any]
    suites: List[TestSuite]

class NISTestRunner:
    """Comprehensive NIS test runner"""
    
    def __init__(self, config_path: str = "testing/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.project_root = self._find_project_root()
        self.test_session_id = self._generate_session_id()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load testing configuration"""
        if not self.config_path.exists():
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'test_runner': {'framework': 'pytest', 'parallel': True, 'max_workers': 4},
            'coverage': {'enabled': True, 'min_threshold': 80.0},
            'security': {'enabled': True, 'tools': ['bandit', 'safety']},
            'quality': {'enabled': True, 'tools': ['flake8', 'black', 'mypy']},
            'benchmarks': {'enabled': True},
            'reporting': {'enabled': True, 'formats': ['html', 'json']}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup test logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('NISTestRunner')
    
    def _find_project_root(self) -> Path:
        """Find project root directory"""
        current = Path.cwd()
        markers = ["nis.config.yaml", "VERSION", "nis-core-toolkit"]
        
        while current != current.parent:
            for marker in markers:
                if (current / marker).exists():
                    return current
            current = current.parent
        
        return Path.cwd()
    
    def _generate_session_id(self) -> str:
        """Generate unique test session ID"""
        return f"test_session_{int(time.time())}"
    
    def run_full_test_suite(self, 
                          test_types: List[str] = None,
                          verbose: bool = False,
                          fail_fast: bool = False) -> TestReport:
        """Run comprehensive test suite"""
        
        if test_types is None:
            test_types = ['unit', 'integration', 'coverage', 'security', 'quality', 'benchmarks']
        
        self.logger.info(f"🧪 Starting comprehensive test suite - Session: {self.test_session_id}")
        start_time = time.time()
        
        # Initialize results
        results = {
            'unit': None,
            'integration': None,
            'coverage': None,
            'security': None,
            'quality': None,
            'benchmarks': None
        }
        
        # Run tests in parallel where possible
        with ThreadPoolExecutor(max_workers=self.config['test_runner'].get('max_workers', 4)) as executor:
            futures = {}
            
            # Submit test jobs
            if 'unit' in test_types:
                futures['unit'] = executor.submit(self._run_unit_tests, verbose, fail_fast)
            
            if 'integration' in test_types and self.config.get('integration', {}).get('enabled', True):
                futures['integration'] = executor.submit(self._run_integration_tests, verbose)
            
            if 'security' in test_types and self.config.get('security', {}).get('enabled', True):
                futures['security'] = executor.submit(self._run_security_scan)
            
            if 'quality' in test_types and self.config.get('quality', {}).get('enabled', True):
                futures['quality'] = executor.submit(self._run_quality_analysis)
            
            # Collect results
            for test_type, future in futures.items():
                try:
                    results[test_type] = future.result()
                    self.logger.info(f"✅ {test_type.title()} tests completed")
                except Exception as e:
                    self.logger.error(f"❌ {test_type.title()} tests failed: {e}")
                    results[test_type] = {'error': str(e)}
        
        # Run coverage analysis (sequential after unit tests)
        if 'coverage' in test_types and self.config.get('coverage', {}).get('enabled', True):
            results['coverage'] = self._run_coverage_analysis()
        
        # Run benchmarks (sequential, resource intensive)
        if 'benchmarks' in test_types and self.config.get('benchmarks', {}).get('enabled', True):
            results['benchmarks'] = self._run_performance_benchmarks()
        
        # Generate comprehensive report
        total_duration = time.time() - start_time
        report = self._generate_test_report(results, total_duration)
        
        # Save and display results
        self._save_test_report(report)
        self._display_test_summary(report)
        
        return report
    
    def _run_unit_tests(self, verbose: bool = False, fail_fast: bool = False) -> Dict[str, Any]:
        """Run unit tests with pytest"""
        self.logger.info("🔬 Running unit tests...")
        
        # Find test directories
        test_dirs = []
        for test_dir in self.config.get('discovery', {}).get('test_directories', ['tests']):
            test_path = self.project_root / test_dir
            if test_path.exists():
                test_dirs.append(str(test_path))
        
        if not test_dirs:
            self.logger.warning("No test directories found")
            return {'status': 'skipped', 'message': 'No tests found'}
        
        # Build pytest command
        cmd = [
            sys.executable, '-m', 'pytest',
            *test_dirs,
            '--tb=short',
            '--junit-xml=testing/reports/junit.xml',
            '--json-report', '--json-report-file=testing/reports/pytest.json'
        ]
        
        if verbose:
            cmd.append('-v')
        
        if fail_fast:
            cmd.append('-x')
        
        if self.config.get('test_runner', {}).get('parallel', True):
            cmd.extend(['-n', str(self.config['test_runner'].get('max_workers', 4))])
        
        # Run tests
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config.get('test_runner', {}).get('timeout', 300)
            )
            
            # Parse results
            return self._parse_pytest_results(result)
            
        except subprocess.TimeoutExpired:
            self.logger.error("Unit tests timed out")
            return {'status': 'timeout', 'message': 'Tests exceeded timeout limit'}
        except Exception as e:
            self.logger.error(f"Unit tests failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests with Docker"""
        self.logger.info("🔗 Running integration tests...")
        
        compose_file = self.config.get('integration', {}).get('docker_compose_file', 'docker-compose.test.yml')
        compose_path = self.project_root / compose_file
        
        if not compose_path.exists():
            return {'status': 'skipped', 'message': 'No integration test configuration found'}
        
        try:
            # Start test environment
            self.logger.info("Starting test environment...")
            subprocess.run([
                'docker', 'compose', '-f', str(compose_path),
                'up', '-d', '--build'
            ], check=True, cwd=self.project_root)
            
            # Wait for services to be ready
            time.sleep(10)
            
            # Run integration tests
            result = subprocess.run([
                'docker', 'compose', '-f', str(compose_path),
                'exec', '-T', 'test-runner',
                'python', '-m', 'pytest', 'tests/integration/', '-v'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'errors': result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            return {'status': 'failed', 'message': f'Integration tests failed: {e}'}
        finally:
            # Cleanup test environment
            try:
                subprocess.run([
                    'docker', 'compose', '-f', str(compose_path),
                    'down', '-v'
                ], cwd=self.project_root)
            except:
                pass
    
    def _run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis"""
        self.logger.info("📊 Running coverage analysis...")
        
        try:
            # Run tests with coverage
            cmd = [
                sys.executable, '-m', 'pytest',
                '--cov=' + ','.join(self.config.get('coverage', {}).get('source_dirs', ['src'])),
                '--cov-report=html:testing/reports/coverage_html',
                '--cov-report=xml:testing/reports/coverage.xml',
                '--cov-report=json:testing/reports/coverage.json',
                '--cov-report=term',
                f"--cov-fail-under={self.config.get('coverage', {}).get('fail_under', 75)}"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Parse coverage results
            coverage_file = self.project_root / 'testing/reports/coverage.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    return {
                        'status': 'passed' if result.returncode == 0 else 'failed',
                        'percentage': coverage_data.get('totals', {}).get('percent_covered', 0),
                        'lines_covered': coverage_data.get('totals', {}).get('covered_lines', 0),
                        'lines_total': coverage_data.get('totals', {}).get('num_statements', 0),
                        'files': len(coverage_data.get('files', {}))
                    }
            
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout
            }
            
        except Exception as e:
            self.logger.error(f"Coverage analysis failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _run_security_scan(self) -> Dict[str, Any]:
        """Run security vulnerability scanning"""
        self.logger.info("🔒 Running security scan...")
        
        results = {'tools': {}, 'total_issues': 0, 'status': 'passed'}
        
        security_tools = self.config.get('security', {}).get('tools', ['bandit', 'safety'])
        
        # Run Bandit (Python security linter)
        if 'bandit' in security_tools:
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'bandit',
                    '-r', 'nis-core-toolkit/src',
                    '-f', 'json',
                    '-o', 'testing/reports/bandit.json'
                ], capture_output=True, text=True, cwd=self.project_root)
                
                results['tools']['bandit'] = {
                    'status': 'passed' if result.returncode == 0 else 'issues_found',
                    'output': result.stdout
                }
            except Exception as e:
                results['tools']['bandit'] = {'status': 'error', 'message': str(e)}
        
        # Run Safety (dependency vulnerability scanner)
        if 'safety' in security_tools:
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'safety', 'check',
                    '--json', '--output', 'testing/reports/safety.json'
                ], capture_output=True, text=True, cwd=self.project_root)
                
                results['tools']['safety'] = {
                    'status': 'passed' if result.returncode == 0 else 'vulnerabilities_found',
                    'output': result.stdout
                }
            except Exception as e:
                results['tools']['safety'] = {'status': 'error', 'message': str(e)}
        
        return results
    
    def _run_quality_analysis(self) -> Dict[str, Any]:
        """Run code quality analysis"""
        self.logger.info("✨ Running code quality analysis...")
        
        results = {'tools': {}, 'overall_score': 0, 'status': 'passed'}
        
        quality_tools = self.config.get('quality', {}).get('tools', ['flake8', 'black', 'mypy'])
        
        # Run flake8 (style and error linting)
        if 'flake8' in quality_tools:
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'flake8',
                    'nis-core-toolkit/src',
                    '--output-file=testing/reports/flake8.txt',
                    '--format=json'
                ], capture_output=True, text=True, cwd=self.project_root)
                
                results['tools']['flake8'] = {
                    'status': 'passed' if result.returncode == 0 else 'issues_found',
                    'issues': result.stdout.count('\n') if result.stdout else 0
                }
            except Exception as e:
                results['tools']['flake8'] = {'status': 'error', 'message': str(e)}
        
        # Run mypy (type checking)
        if 'mypy' in quality_tools:
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'mypy',
                    'nis-core-toolkit/src',
                    '--json-report', 'testing/reports/mypy.json'
                ], capture_output=True, text=True, cwd=self.project_root)
                
                results['tools']['mypy'] = {
                    'status': 'passed' if result.returncode == 0 else 'type_errors',
                    'output': result.stdout
                }
            except Exception as e:
                results['tools']['mypy'] = {'status': 'error', 'message': str(e)}
        
        return results
    
    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        self.logger.info("⚡ Running performance benchmarks...")
        
        try:
            # Run pytest with benchmark plugin
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'testing/benchmarks/',
                '--benchmark-only',
                '--benchmark-json=testing/reports/benchmarks.json',
                '--benchmark-histogram=testing/reports/benchmark_histogram'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Parse benchmark results
            benchmark_file = self.project_root / 'testing/reports/benchmarks.json'
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                    return {
                        'status': 'passed' if result.returncode == 0 else 'failed',
                        'benchmarks': benchmark_data.get('benchmarks', []),
                        'machine_info': benchmark_data.get('machine_info', {}),
                        'commit_info': benchmark_data.get('commit_info', {})
                    }
            
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout
            }
            
        except Exception as e:
            self.logger.error(f"Benchmarks failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _parse_pytest_results(self, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """Parse pytest results"""
        try:
            # Try to load JSON report
            json_report_path = self.project_root / 'testing/reports/pytest.json'
            if json_report_path.exists():
                with open(json_report_path) as f:
                    data = json.load(f)
                    return {
                        'status': 'passed' if result.returncode == 0 else 'failed',
                        'total': data.get('summary', {}).get('total', 0),
                        'passed': data.get('summary', {}).get('passed', 0),
                        'failed': data.get('summary', {}).get('failed', 0),
                        'skipped': data.get('summary', {}).get('skipped', 0),
                        'duration': data.get('duration', 0),
                        'tests': data.get('tests', [])
                    }
            
            # Fallback to parsing stdout
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'errors': result.stderr
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Failed to parse test results: {e}",
                'output': result.stdout,
                'errors': result.stderr
            }
    
    def _generate_test_report(self, results: Dict[str, Any], total_duration: float) -> TestReport:
        """Generate comprehensive test report"""
        
        # Extract metrics from results
        unit_results = results.get('unit') or {}
        coverage_results = results.get('coverage') or {}
        security_results = results.get('security') or {}
        quality_results = results.get('quality') or {}
        benchmark_results = results.get('benchmarks') or {}
        
        return TestReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            total_tests=unit_results.get('total', 0),
            passed=unit_results.get('passed', 0),
            failed=unit_results.get('failed', 0),
            skipped=unit_results.get('skipped', 0),
            errors=0,  # Will be computed from detailed results
            total_duration=total_duration,
            coverage_percentage=coverage_results.get('percentage', 0),
            security_issues=security_results.get('total_issues', 0),
            quality_score=quality_results.get('overall_score', 0),
            performance_metrics=benchmark_results,
            suites=[]  # Will be populated from detailed results
        )
    
    def _save_test_report(self, report: TestReport):
        """Save test report to file"""
        reports_dir = self.project_root / 'testing/reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        report_file = reports_dir / f'test_report_{self.test_session_id}.json'
        with open(report_file, 'w') as f:
            json.dump(report.__dict__, f, indent=2, default=str)
        
        # Save as HTML (basic template)
        html_report = self._generate_html_report(report)
        html_file = reports_dir / f'test_report_{self.test_session_id}.html'
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        self.logger.info(f"📋 Test report saved: {report_file}")
        self.logger.info(f"🌐 HTML report: {html_file}")
    
    def _generate_html_report(self, report: TestReport) -> str:
        """Generate HTML report"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NIS Test Report - {report.timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
                .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; flex: 1; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .coverage {{ background: linear-gradient(90deg, green {report.coverage_percentage}%, #f0f0f0 {report.coverage_percentage}%); }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 NIS TOOLKIT SUIT - Test Report</h1>
                <p>Generated: {report.timestamp}</p>
                <p>Duration: {report.total_duration:.2f} seconds</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>Test Results</h3>
                    <p class="passed">✅ Passed: {report.passed}</p>
                    <p class="failed">❌ Failed: {report.failed}</p>
                    <p>⏭️ Skipped: {report.skipped}</p>
                    <p><strong>Total: {report.total_tests}</strong></p>
                </div>
                
                <div class="metric">
                    <h3>Coverage</h3>
                    <div class="coverage" style="height: 20px; border-radius: 10px;"></div>
                    <p><strong>{report.coverage_percentage:.1f}%</strong></p>
                </div>
                
                <div class="metric">
                    <h3>Security</h3>
                    <p>Issues: {report.security_issues}</p>
                    <p>Status: {'✅ Clean' if report.security_issues == 0 else '⚠️ Issues Found'}</p>
                </div>
            </div>
            
            <div class="metric">
                <h3>Performance Benchmarks</h3>
                <p>Benchmark results available in detailed JSON report</p>
            </div>
        </body>
        </html>
        """
    
    def _display_test_summary(self, report: TestReport):
        """Display test summary in console"""
        print("\n" + "="*60)
        print("🧪 NIS TOOLKIT SUIT - Test Summary")
        print("="*60)
        print(f"📅 Timestamp: {report.timestamp}")
        print(f"⏱️  Duration: {report.total_duration:.2f}s")
        print()
        
        # Test results
        print("📊 Test Results:")
        print(f"   ✅ Passed: {report.passed}")
        print(f"   ❌ Failed: {report.failed}")
        print(f"   ⏭️  Skipped: {report.skipped}")
        print(f"   📈 Total: {report.total_tests}")
        print()
        
        # Coverage
        if report.coverage_percentage > 0:
            print(f"📊 Coverage: {report.coverage_percentage:.1f}%")
            coverage_bar = "█" * int(report.coverage_percentage / 2.5) + "░" * (40 - int(report.coverage_percentage / 2.5))
            print(f"   [{coverage_bar}]")
            print()
        
        # Security
        if report.security_issues == 0:
            print("🔒 Security: ✅ No issues found")
        else:
            print(f"🔒 Security: ⚠️  {report.security_issues} issues found")
        print()
        
        # Overall status
        if report.failed == 0 and report.security_issues == 0:
            print("🎉 Overall Status: ✅ ALL TESTS PASSED")
        else:
            print("⚠️  Overall Status: ❌ ISSUES FOUND")
        
        print("="*60)


def main():
    """Main test runner entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NIS Comprehensive Test Runner')
    parser.add_argument('--config', '-c', help='Test configuration file')
    parser.add_argument('--types', '-t', nargs='+', 
                       choices=['unit', 'integration', 'coverage', 'security', 'quality', 'benchmarks'],
                       default=['unit', 'coverage', 'security', 'quality'],
                       help='Test types to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fail-fast', '-x', action='store_true', help='Stop on first failure')
    
    args = parser.parse_args()
    
    # Create test runner
    config_path = args.config or "testing/config.yaml"
    runner = NISTestRunner(config_path)
    
    # Run tests
    report = runner.run_full_test_suite(
        test_types=args.types,
        verbose=args.verbose,
        fail_fast=args.fail_fast
    )
    
    # Exit with appropriate code
    exit_code = 0 if report.failed == 0 else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
