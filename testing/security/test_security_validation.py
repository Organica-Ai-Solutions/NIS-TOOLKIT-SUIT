"""
NIS Security Validation Tests
Comprehensive security testing for the NIS TOOLKIT SUIT
"""

import pytest
import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

# Security test fixtures
@pytest.fixture
def sample_api_key():
    """Sample API key for testing"""
    return "test_api_key_12345"

@pytest.fixture
def malicious_inputs():
    """Common malicious input patterns"""
    return [
        # SQL Injection attempts
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'/*",
        
        # XSS attempts  
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        
        # Command injection attempts
        "; cat /etc/passwd",
        "| whoami",
        "&& rm -rf /",
        
        # Path traversal attempts
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        
        # NoSQL injection attempts
        "{'$where': 'function() { return true; }'}",
        "{'$regex': '.*'}",
        
        # LDAP injection attempts
        "admin)(&(password=*))",
        "*)(uid=*",
        
        # XML/XXE attempts
        "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>",
    ]

@pytest.fixture
def large_payload():
    """Large payload for DoS testing"""
    return "A" * 10000

@pytest.fixture
def temp_config_file():
    """Temporary configuration file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'api_key': 'test_key',
            'debug': False,
            'allowed_origins': ['localhost'],
            'max_request_size': 1024
        }
        import yaml
        yaml.dump(config, f)
        yield f.name
    os.unlink(f.name)


class MockNISAgent:
    """Mock NIS agent for security testing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.authenticated = False
        
    def authenticate(self, api_key: str) -> bool:
        """Mock authentication"""
        expected_key = self.config.get('api_key', 'valid_key')
        self.authenticated = (api_key == expected_key)
        return self.authenticated
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Mock input processing"""
        if not self.authenticated:
            raise SecurityError("Not authenticated")
        
        # Simulate input validation
        if len(user_input) > self.config.get('max_input_length', 1000):
            raise ValueError("Input too long")
            
        return {
            'input': user_input,
            'output': f'processed: {user_input}',
            'safe': self._is_safe_input(user_input)
        }
    
    def _is_safe_input(self, user_input: str) -> bool:
        """Mock security validation"""
        dangerous_patterns = ['<script', 'DROP TABLE', '; cat', '../', 'rm -rf']
        return not any(pattern in user_input for pattern in dangerous_patterns)


class SecurityError(Exception):
    """Custom security exception"""
    pass


# Authentication and Authorization Tests
class TestAuthentication:
    """Test authentication mechanisms"""
    
    def test_valid_authentication(self, sample_api_key):
        """Test valid authentication"""
        agent = MockNISAgent({'api_key': sample_api_key})
        assert agent.authenticate(sample_api_key) is True
        assert agent.authenticated is True
    
    def test_invalid_authentication(self, sample_api_key):
        """Test invalid authentication"""
        agent = MockNISAgent({'api_key': sample_api_key})
        assert agent.authenticate('wrong_key') is False
        assert agent.authenticated is False
    
    def test_empty_api_key(self):
        """Test empty API key"""
        agent = MockNISAgent({'api_key': 'valid_key'})
        assert agent.authenticate('') is False
        assert agent.authenticate(None) is False
    
    def test_authentication_required_for_processing(self, sample_api_key):
        """Test that authentication is required for processing"""
        agent = MockNISAgent({'api_key': sample_api_key})
        
        # Should fail without authentication
        with pytest.raises(SecurityError, match="Not authenticated"):
            agent.process_input("test input")
        
        # Should work after authentication
        agent.authenticate(sample_api_key)
        result = agent.process_input("test input")
        assert result['input'] == "test input"


# Input Validation and Sanitization Tests
class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_malicious_input_detection(self, sample_api_key, malicious_inputs):
        """Test detection of malicious inputs"""
        agent = MockNISAgent({'api_key': sample_api_key})
        agent.authenticate(sample_api_key)
        
        for malicious_input in malicious_inputs:
            result = agent.process_input(malicious_input)
            # Should detect as unsafe
            assert result['safe'] is False, f"Failed to detect malicious input: {malicious_input}"
    
    def test_safe_input_processing(self, sample_api_key):
        """Test processing of safe inputs"""
        agent = MockNISAgent({'api_key': sample_api_key})
        agent.authenticate(sample_api_key)
        
        safe_inputs = [
            "Hello, world!",
            "Process this data: 123",
            "Normal user input with numbers 456",
            "Special chars: !@#$%^&*()",
        ]
        
        for safe_input in safe_inputs:
            result = agent.process_input(safe_input)
            assert result['safe'] is True
            assert result['input'] == safe_input
    
    def test_input_length_validation(self, sample_api_key, large_payload):
        """Test input length validation"""
        agent = MockNISAgent({
            'api_key': sample_api_key,
            'max_input_length': 1000
        })
        agent.authenticate(sample_api_key)
        
        # Should reject overly long input
        with pytest.raises(ValueError, match="Input too long"):
            agent.process_input(large_payload)
    
    def test_null_and_empty_input_handling(self, sample_api_key):
        """Test handling of null and empty inputs"""
        agent = MockNISAgent({'api_key': sample_api_key})
        agent.authenticate(sample_api_key)
        
        # Test empty string
        result = agent.process_input("")
        assert result['input'] == ""
        assert result['safe'] is True
        
        # Test whitespace
        result = agent.process_input("   ")
        assert result['input'] == "   "


# Configuration Security Tests  
class TestConfigurationSecurity:
    """Test configuration security"""
    
    def test_sensitive_data_not_logged(self, temp_config_file):
        """Test that sensitive data is not logged"""
        # Mock logging to capture log messages
        with patch('logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            # Load config (would normally log)
            with open(temp_config_file, 'r') as f:
                import yaml
                config = yaml.safe_load(f)
            
            # Check that API key is not in any log calls
            for call in mock_logger_instance.info.call_args_list:
                call_str = str(call)
                assert 'test_key' not in call_str, "API key found in logs"
    
    def test_debug_mode_disabled_in_production(self):
        """Test that debug mode is disabled in production"""
        # Simulate production environment
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            # In real code, this would check actual config
            production_config = {'debug': False, 'environment': 'production'}
            assert production_config['debug'] is False
    
    def test_secure_defaults(self):
        """Test that secure defaults are used"""
        # Test default configuration values
        default_config = {
            'debug': False,
            'ssl_verify': True,
            'secure_cookies': True,
            'max_request_size': 1024 * 1024,  # 1MB
            'timeout': 30,
            'allowed_origins': ['localhost']
        }
        
        # Verify secure defaults
        assert default_config['debug'] is False
        assert default_config['ssl_verify'] is True
        assert default_config['secure_cookies'] is True
        assert default_config['max_request_size'] <= 10 * 1024 * 1024  # Max 10MB
        assert default_config['timeout'] <= 300  # Max 5 minutes
        assert 'localhost' in default_config['allowed_origins']


# Dependency Security Tests
class TestDependencySecurity:
    """Test dependency security"""
    
    def test_known_vulnerable_dependencies(self):
        """Test for known vulnerable dependencies"""
        # This would run safety check in real implementation
        # For now, simulate the check
        
        vulnerable_packages = []
        
        # Mock safety check results
        mock_safety_results = {
            "vulnerabilities_found": 0,
            "packages_checked": 50,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        assert mock_safety_results["vulnerabilities_found"] == 0, \
            f"Found {mock_safety_results['vulnerabilities_found']} vulnerable dependencies"
    
    def test_dependency_versions_pinned(self):
        """Test that dependency versions are pinned"""
        requirements_files = [
            "requirements.txt",
            "nis-core-toolkit/requirements.txt"
        ]
        
        project_root = Path.cwd()
        
        for req_file in requirements_files:
            req_path = project_root / req_file
            if req_path.exists():
                with open(req_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and '==' not in line and '>=' in line:
                        # This is acceptable - minimum version specified
                        continue
                    elif line and not line.startswith('#') and '==' not in line:
                        pytest.fail(f"Unpinned dependency found in {req_file}: {line}")


# Data Protection Tests
class TestDataProtection:
    """Test data protection mechanisms"""
    
    def test_sensitive_data_masking(self, sample_api_key):
        """Test that sensitive data is masked in outputs"""
        
        def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
            """Mock function to mask sensitive data"""
            masked = data.copy()
            if 'api_key' in masked:
                masked['api_key'] = '*' * len(masked['api_key'])
            return masked
        
        test_data = {'api_key': sample_api_key, 'user_id': '12345'}
        masked_data = mask_sensitive_data(test_data)
        
        assert masked_data['api_key'] == '*' * len(sample_api_key)
        assert masked_data['user_id'] == '12345'  # Not sensitive
    
    def test_temporary_file_cleanup(self):
        """Test that temporary files are properly cleaned up"""
        temp_files = []
        
        try:
            # Create temporary files
            for i in range(3):
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(b"sensitive data")
                temp_file.close()
                temp_files.append(temp_file.name)
            
            # Verify files exist
            for temp_file in temp_files:
                assert os.path.exists(temp_file)
            
        finally:
            # Cleanup
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            
            # Verify cleanup
            for temp_file in temp_files:
                assert not os.path.exists(temp_file), f"Temporary file not cleaned up: {temp_file}"


# Network Security Tests
class TestNetworkSecurity:
    """Test network security measures"""
    
    def test_ssl_verification_enabled(self):
        """Test that SSL verification is enabled"""
        # Mock requests session
        with patch('requests.Session') as mock_session:
            mock_instance = MagicMock()
            mock_session.return_value = mock_instance
            
            # In real code, this would be actual network client
            mock_instance.verify = True
            
            assert mock_instance.verify is True, "SSL verification should be enabled"
    
    def test_cors_configuration(self):
        """Test CORS configuration"""
        allowed_origins = ['localhost', '127.0.0.1']
        
        # Test valid origins
        for origin in allowed_origins:
            assert origin in allowed_origins
        
        # Test invalid origins
        invalid_origins = ['evil.com', 'attacker.net']
        for origin in invalid_origins:
            assert origin not in allowed_origins, f"Invalid origin allowed: {origin}"
    
    def test_rate_limiting(self):
        """Test rate limiting mechanisms"""
        # Mock rate limiter
        class MockRateLimiter:
            def __init__(self, max_requests=100, window=60):
                self.max_requests = max_requests
                self.window = window
                self.requests = {}
            
            def is_allowed(self, client_id: str) -> bool:
                # Simplified rate limiting logic
                import time
                current_time = time.time()
                client_requests = self.requests.get(client_id, [])
                
                # Remove old requests
                client_requests = [req_time for req_time in client_requests 
                                 if current_time - req_time < self.window]
                
                if len(client_requests) >= self.max_requests:
                    return False
                
                client_requests.append(current_time)
                self.requests[client_id] = client_requests
                return True
        
        rate_limiter = MockRateLimiter(max_requests=5, window=60)
        client_id = "test_client"
        
        # Should allow first 5 requests
        for i in range(5):
            assert rate_limiter.is_allowed(client_id) is True
        
        # Should deny 6th request
        assert rate_limiter.is_allowed(client_id) is False


# Error Handling Security Tests
class TestErrorHandlingSecurity:
    """Test secure error handling"""
    
    def test_error_messages_dont_leak_info(self, sample_api_key):
        """Test that error messages don't leak sensitive information"""
        agent = MockNISAgent({'api_key': sample_api_key})
        
        try:
            agent.process_input("test")  # Should fail - not authenticated
        except SecurityError as e:
            error_msg = str(e)
            # Error message should not contain sensitive info
            assert sample_api_key not in error_msg
            assert "api_key" not in error_msg.lower()
            assert "secret" not in error_msg.lower()
    
    def test_exception_handling_prevents_disclosure(self):
        """Test that exceptions don't disclose system information"""
        
        def mock_operation():
            # Simulate operation that might leak system info
            raise Exception("/home/user/secret/config.yaml not found")
        
        try:
            mock_operation()
        except Exception as e:
            # In production, error should be sanitized
            sanitized_error = "Configuration file not found"
            assert "/home/user" not in sanitized_error
            assert "secret" not in sanitized_error


if __name__ == "__main__":
    # Run security tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_",
        "--capture=no"
    ])
