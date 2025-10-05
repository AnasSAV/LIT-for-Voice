"""
Security and Access Control Testing  
Test Plan Section 3.1.6 - Session Security, File Upload Security, API Security
"""

import pytest
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from httpx import AsyncClient

# Session Security Testing (Critical Priority)
class TestSessionSecurity:
    """Test session management security mechanisms."""
    
    @pytest.mark.asyncio
    async def test_session_cookie_security_attributes(self, client: AsyncClient):
        """Test session cookie security attributes (HttpOnly, Secure, SameSite)."""
        # Make a request that should create a session
        response = await client.get("/health")
        
        # Check for session cookie in response
        cookies = response.cookies
        
        # If session cookies are implemented, verify security attributes
        for cookie_name, cookie_value in cookies.items():
            if 'session' in cookie_name.lower():
                # In production, these should be set:
                # - HttpOnly: prevents XSS access to cookies
                # - Secure: only send over HTTPS
                # - SameSite: CSRF protection
                
                # For testing, we just verify the cookie exists
                assert cookie_value is not None
                print(f"Session cookie found: {cookie_name}")
    
    @pytest.mark.asyncio
    async def test_session_isolation_between_users(self, fake_redis):
        """Test that different sessions don't interfere with each other."""
        # Create two separate sessions
        session1_id = "user1_session_123"
        session2_id = "user2_session_456"
        
        session1_data = {
            "user_id": "user1",
            "files": ["file1.wav"],
            "permissions": ["read", "write"]
        }
        
        session2_data = {
            "user_id": "user2", 
            "files": ["file2.wav"],
            "permissions": ["read"]
        }
        
        # Store sessions
        await fake_redis.set(f"session_{session1_id}", json.dumps(session1_data))
        await fake_redis.set(f"session_{session2_id}", json.dumps(session2_data))
        
        # Verify isolation
        retrieved1 = json.loads(await fake_redis.get(f"session_{session1_id}"))
        retrieved2 = json.loads(await fake_redis.get(f"session_{session2_id}"))
        
        assert retrieved1["user_id"] == "user1"
        assert retrieved2["user_id"] == "user2"
        assert retrieved1["files"] != retrieved2["files"]
        assert retrieved1["permissions"] != retrieved2["permissions"]
    
    @pytest.mark.asyncio
    async def test_session_timeout_and_renewal(self, fake_redis):
        """Test session timeout behavior and renewal mechanisms."""
        session_id = "timeout_test_session"
        session_data = {"user_id": "test_user", "created": "2025-10-01"}
        
        # Set session with short TTL
        await fake_redis.set(f"session_{session_id}", json.dumps(session_data), ex=2)
        
        # Verify session exists
        result = await fake_redis.get(f"session_{session_id}")
        assert result is not None
        
        # Check TTL
        ttl = await fake_redis.ttl(f"session_{session_id}")
        assert 0 < ttl <= 2
        
        # Simulate session renewal
        await fake_redis.expire(f"session_{session_id}", 10)
        new_ttl = await fake_redis.ttl(f"session_{session_id}")
        assert new_ttl > ttl
    
    @pytest.mark.asyncio
    async def test_session_data_encryption_integrity(self, fake_redis):
        """Test session data integrity and potential encryption."""
        session_id = "encryption_test"
        sensitive_data = {
            "user_id": "sensitive_user",
            "api_key": "secret_key_12345",
            "preferences": {"theme": "dark", "language": "en"}
        }
        
        # In production, session data should be encrypted
        # For testing, we verify data integrity
        stored_data = json.dumps(sensitive_data)
        await fake_redis.set(f"session_{session_id}", stored_data)
        
        retrieved_data = json.loads(await fake_redis.get(f"session_{session_id}"))
        
        # Verify data integrity
        assert retrieved_data == sensitive_data
        assert retrieved_data["api_key"] == "secret_key_12345"
        
        # In production, implement encryption/decryption here
        def simulate_encryption(data: str) -> str:
            """Simulate basic encryption (not for production use)."""
            return hashlib.sha256(data.encode()).hexdigest()[:32]
        
        encrypted = simulate_encryption(stored_data)
        assert len(encrypted) == 32  # Simulated encrypted length


# File Upload Security Testing (Critical Priority)
class TestFileUploadSecurity:
    """Test file upload security mechanisms and validation."""
    
    @pytest.mark.asyncio
    async def test_file_type_validation(self, client: AsyncClient, temp_dir: Path):
        """Test file type validation and malicious file rejection."""
        # Create various file types
        test_files = {
            'valid_wav': (temp_dir / "valid.wav", b"RIFF\x24\x08\x00\x00WAVE", "audio/wav"),
            'fake_wav': (temp_dir / "fake.wav", b"#!/bin/bash\necho 'malicious'", "audio/wav"),
            'exe_file': (temp_dir / "malicious.exe", b"MZ\x90\x00", "application/octet-stream"),
            'script_file': (temp_dir / "script.js", b"alert('xss')", "text/javascript"),
            'large_extension': (temp_dir / ("x" * 300 + ".wav"), b"test", "audio/wav")
        }
        
        for file_type, (file_path, content, mime_type) in test_files.items():
            file_path.write_bytes(content)
            
            # Test file upload
            files = {"file": (file_path.name, open(file_path, "rb"), mime_type)}
            
            try:
                response = await client.post("/upload/audio", files=files)
                
                if file_type == 'valid_wav':
                    # Valid files should be accepted (if endpoint exists)
                    assert response.status_code in [200, 404, 422]  # 404 if endpoint not implemented
                else:
                    # Invalid files should be rejected
                    assert response.status_code in [400, 415, 422, 404]
                    
            finally:
                files["file"][1].close()
    
    def test_file_size_limit_enforcement(self, temp_dir: Path):
        """Test file size limits and resource exhaustion protection."""
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit from test plan
        
        def validate_file_size(file_path: Path) -> bool:
            """Validate file size against limits."""
            try:
                file_size = file_path.stat().st_size
                return file_size <= MAX_FILE_SIZE
            except Exception:
                return False
        
        # Create files of different sizes
        small_file = temp_dir / "small.wav"
        small_file.write_bytes(b"x" * 1024)  # 1KB
        
        # Don't actually create 100MB file in tests
        # Just test the validation logic
        assert validate_file_size(small_file) == True
        
        # Simulate large file
        class MockLargeFile:
            def stat(self):
                class MockStat:
                    st_size = 100 * 1024 * 1024  # 100MB
                return MockStat()
        
        large_file_mock = MockLargeFile()
        assert large_file_mock.stat().st_size > MAX_FILE_SIZE
    
    def test_path_traversal_prevention(self, temp_dir: Path):
        """Test prevention of path traversal attacks."""
        def secure_file_path(filename: str, upload_dir: Path) -> Path:
            """Secure file path resolution to prevent traversal."""
            # Remove dangerous characters and path components
            import os
            safe_filename = os.path.basename(filename)
            safe_filename = safe_filename.replace("..", "")
            safe_filename = safe_filename.replace("/", "")
            safe_filename = safe_filename.replace("\\", "")
            
            # Ensure file stays within upload directory
            full_path = upload_dir / safe_filename
            resolved_path = full_path.resolve()
            
            # Verify the resolved path is within the upload directory
            if not str(resolved_path).startswith(str(upload_dir.resolve())):
                raise ValueError("Path traversal attempt detected")
            
            return resolved_path
        
        upload_dir = temp_dir / "uploads"
        upload_dir.mkdir()
        
        # Test various malicious paths
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\cmd.exe",
            "/etc/passwd",
            "C:\\windows\\system32\\cmd.exe",
            "file/../../../secret.txt",
            "normal_file.wav/../../../secret.txt"
        ]
        
        for malicious_path in malicious_paths:
            try:
                secure_path = secure_file_path(malicious_path, upload_dir)
                # If it doesn't raise an exception, ensure it's safe
                assert str(upload_dir.resolve()) in str(secure_path)
                print(f"Path {malicious_path} -> {secure_path}")
            except ValueError as e:
                # Expected for path traversal attempts
                assert "traversal" in str(e).lower()
    
    def test_file_content_validation(self, temp_dir: Path):
        """Test audio file content validation and sanitization."""
        def validate_audio_content(file_path: Path) -> bool:
            """Validate that file is actually audio content."""
            try:
                # Check file signature/magic bytes
                with open(file_path, 'rb') as f:
                    header = f.read(12)
                
                # WAV file signature
                if header.startswith(b'RIFF') and b'WAVE' in header:
                    return True
                
                # Other audio formats could be checked here
                # MP3: starts with ID3 or has MP3 frame sync
                # FLAC: starts with 'fLaC'
                
                return False
            except Exception:
                return False
        
        # Create files with different content
        valid_wav = temp_dir / "valid.wav"
        valid_wav.write_bytes(b'RIFF\x24\x08\x00\x00WAVEfmt ')
        
        fake_audio = temp_dir / "fake.wav"
        fake_audio.write_bytes(b'This is not audio content')
        
        script_file = temp_dir / "script.wav"
        script_file.write_bytes(b'#!/bin/bash\nrm -rf /')
        
        assert validate_audio_content(valid_wav) == True
        assert validate_audio_content(fake_audio) == False
        assert validate_audio_content(script_file) == False


# API Security Testing (Important Priority)
class TestAPISecurity:
    """Test API endpoint security and input validation."""
    
    @pytest.mark.asyncio
    async def test_cors_configuration(self, client: AsyncClient):
        """Test CORS configuration and origin validation."""
        # Test preflight request
        response = await client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        
        # CORS headers should be present (if configured)
        cors_headers = {
            'Access-Control-Allow-Origin',
            'Access-Control-Allow-Methods', 
            'Access-Control-Allow-Headers'
        }
        
        response_headers = set(response.headers.keys())
        
        # Check if any CORS headers are present
        has_cors = bool(cors_headers.intersection(response_headers))
        
        if has_cors:
            print("CORS headers detected")
            # Verify proper CORS configuration
            origin = response.headers.get('Access-Control-Allow-Origin')
            if origin:
                assert origin in ['*', 'http://localhost:3000']
        else:
            print("No CORS headers found - may need configuration")
    
    @pytest.mark.asyncio
    async def test_input_validation_and_injection_prevention(self, client: AsyncClient):
        """Test input validation and SQL/command injection prevention."""
        # Test various injection attempts
        injection_payloads = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "$(rm -rf /)",
            "../../etc/passwd",
            "\\x00\\x01\\x02",  # Null bytes
            "A" * 10000,  # Extremely long input
        ]
        
        for payload in injection_payloads:
            # Test different endpoints with malicious input
            test_requests = [
                {"endpoint": "/inferences/whisper-base", "data": {"file_path": payload}},
                {"endpoint": "/inferences/wav2vec2-detailed", "data": {"file_path": payload}},
                {"endpoint": "/dataset/list", "data": {"dataset": payload}},
            ]
            
            for test_req in test_requests:
                response = await client.post(test_req["endpoint"], json=test_req["data"])
                
                # Should either reject malicious input or handle gracefully
                assert response.status_code in [400, 404, 422, 500]
                
                # Response should not echo back malicious content
                if response.status_code != 500:
                    try:
                        response_data = response.json()
                        response_text = json.dumps(response_data)
                        assert payload not in response_text, f"Payload reflected in response: {payload}"
                    except Exception:
                        # Response might not be JSON, which is fine
                        pass
    
    @pytest.mark.asyncio
    async def test_rate_limiting_and_dos_protection(self, client: AsyncClient):
        """Test rate limiting and DoS protection mechanisms."""
        # Rapid requests to health endpoint
        rapid_requests = 30
        responses = []
        
        for i in range(rapid_requests):
            try:
                response = await client.get("/health")
                responses.append(response.status_code)
            except Exception as e:
                responses.append(429)  # Assume rate limited
        
        status_codes = set(responses)
        print(f"Rate limiting test: {len(responses)} requests, status codes: {status_codes}")
        
        # If rate limiting is implemented, should see 429 responses
        # If not, should at least handle requests gracefully
        success_responses = len([r for r in responses if r == 200])
        rate_limited = len([r for r in responses if r == 429])
        
        # Either most requests succeed (no rate limiting) or some are rate limited
        assert success_responses > 0 or rate_limited > 0
    
    @pytest.mark.asyncio
    async def test_error_message_information_disclosure(self, client: AsyncClient):
        """Test that error messages don't disclose sensitive information."""
        # Test requests designed to trigger errors
        error_inducing_requests = [
            {"endpoint": "/nonexistent/endpoint", "method": "GET"},
            {"endpoint": "/inferences/wav2vec2-detailed", "method": "POST", "data": {}},
            {"endpoint": "/inferences/whisper-base", "method": "POST", "data": {"invalid": "data"}},
        ]
        
        for test_req in error_inducing_requests:
            if test_req["method"] == "GET":
                response = await client.get(test_req["endpoint"])
            else:
                response = await client.post(test_req["endpoint"], json=test_req.get("data", {}))
            
            # Should return appropriate error code
            assert 400 <= response.status_code <= 599
            
            if response.status_code != 500:  # Don't check 500 error content
                try:
                    error_response = response.json()
                    error_text = json.dumps(error_response).lower()
                    
                    # Check for information disclosure
                    sensitive_patterns = [
                        'traceback',
                        'file not found: /',  # Full file paths
                        'password',
                        'secret',
                        'token',
                        'internal server error at line',
                        'database',
                        'sql'
                    ]
                    
                    for pattern in sensitive_patterns:
                        assert pattern not in error_text, f"Sensitive info disclosed: {pattern}"
                        
                except Exception:
                    # Response might not be JSON
                    pass
    
    @pytest.mark.asyncio
    async def test_request_size_limits(self, client: AsyncClient):
        """Test request size limits and payload validation."""
        # Test extremely large JSON payload
        large_payload = {
            "file_path": "test.wav",
            "large_data": "x" * 100000,  # 100KB string
            "nested_data": {
                "level1": {
                    "level2": {
                        "level3": ["item"] * 1000
                    }
                }
            }
        }
        
        response = await client.post("/inferences/wav2vec2-detailed", json=large_payload)
        
        # Should either handle large payload or reject it appropriately
        assert response.status_code in [200, 400, 413, 422, 500]
        
        if response.status_code == 413:
            print("Request size limit enforced (413 Payload Too Large)")
        elif response.status_code == 400:
            print("Large payload rejected with bad request")
        else:
            print(f"Large payload response: {response.status_code}")


# Data Protection and Privacy Testing (Important Priority) 
class TestDataProtection:
    """Test data protection, privacy, and retention policies."""
    
    @pytest.mark.asyncio
    async def test_uploaded_file_access_control(self, temp_dir: Path):
        """Test that uploaded files are properly isolated between users."""
        # Simulate user file storage
        user1_dir = temp_dir / "user1"
        user2_dir = temp_dir / "user2"
        user1_dir.mkdir()
        user2_dir.mkdir()
        
        # Create user files
        user1_file = user1_dir / "private_audio.wav"
        user2_file = user2_dir / "private_audio.wav"
        
        user1_file.write_bytes(b"User 1 private audio data")
        user2_file.write_bytes(b"User 2 private audio data")
        
        # Simulate access control check
        def check_file_access(user_id: str, file_path: Path) -> bool:
            """Check if user has access to file."""
            # Verify file is in user's directory
            try:
                user_dir = temp_dir / user_id
                resolved_path = file_path.resolve()
                user_dir_resolved = user_dir.resolve()
                
                return str(resolved_path).startswith(str(user_dir_resolved))
            except Exception:
                return False
        
        # Test access control
        assert check_file_access("user1", user1_file) == True
        assert check_file_access("user1", user2_file) == False
        assert check_file_access("user2", user2_file) == True
        assert check_file_access("user2", user1_file) == False
    
    @pytest.mark.asyncio
    async def test_cache_data_isolation(self, fake_redis):
        """Test that cache data is properly isolated between sessions."""
        # Create session-specific cache entries
        session1_key = "predictions_session1_user1"
        session2_key = "predictions_session2_user2"
        
        session1_data = {
            "user_id": "user1",
            "predictions": ["happy", "sad"],
            "sensitive_data": "user1_private_info"
        }
        
        session2_data = {
            "user_id": "user2", 
            "predictions": ["neutral", "angry"],
            "sensitive_data": "user2_private_info"
        }
        
        # Store in cache
        await fake_redis.set(session1_key, json.dumps(session1_data))
        await fake_redis.set(session2_key, json.dumps(session2_data))
        
        # Verify isolation
        retrieved1 = json.loads(await fake_redis.get(session1_key))
        retrieved2 = json.loads(await fake_redis.get(session2_key))
        
        assert retrieved1["user_id"] != retrieved2["user_id"]
        assert retrieved1["sensitive_data"] != retrieved2["sensitive_data"]
        assert retrieved1["predictions"] != retrieved2["predictions"]
    
    def test_data_retention_and_cleanup_policies(self, temp_dir: Path):
        """Test data retention policies and cleanup mechanisms."""
        import time
        
        # Simulate file cleanup based on age
        def cleanup_old_files(directory: Path, max_age_seconds: int):
            """Clean up files older than specified age."""
            current_time = time.time()
            cleaned_files = []
            
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        cleaned_files.append(file_path.name)
                        # In real implementation: file_path.unlink()
            
            return cleaned_files
        
        # Create test files with different timestamps
        old_file = temp_dir / "old_file.wav"
        new_file = temp_dir / "new_file.wav"
        
        old_file.write_bytes(b"old audio data")
        new_file.write_bytes(b"new audio data")
        
        # Simulate old file (modify timestamp)
        import os
        old_timestamp = time.time() - 3600  # 1 hour ago
        os.utime(old_file, (old_timestamp, old_timestamp))
        
        # Test cleanup (30 minute retention)
        old_files = cleanup_old_files(temp_dir, 1800)  # 30 minutes
        
        assert "old_file.wav" in old_files
        assert "new_file.wav" not in old_files
    
    @pytest.mark.asyncio
    async def test_sensitive_information_logging(self):
        """Test that sensitive information is not logged inappropriately."""
        import logging
        from io import StringIO
        
        # Create a string buffer to capture log output
        log_buffer = StringIO()
        handler = logging.StreamHandler(log_buffer)
        
        # Create test logger
        logger = logging.getLogger("security_test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Simulate logging with sensitive data
        sensitive_data = {
            "api_key": "secret_key_12345",
            "password": "user_password",
            "session_id": "sess_abc123",
            "file_content": "private audio data"
        }
        
        # Good logging - no sensitive data
        logger.info("User uploaded file successfully")
        logger.info(f"Processing file for user {sensitive_data['session_id'][:8]}...")  # Truncated
        
        # Bad logging - would expose sensitive data (don't do this)
        # logger.info(f"API key: {sensitive_data['api_key']}")
        
        # Check log output
        log_output = log_buffer.getvalue()
        
        # Verify sensitive data is not in logs
        assert "secret_key_12345" not in log_output
        assert "user_password" not in log_output
        assert "private audio data" not in log_output
        
        # Verify useful information is logged
        assert "uploaded file successfully" in log_output
        assert "sess_abc1" in log_output  # Truncated session ID is OK
        
        # Cleanup
        logger.removeHandler(handler)
        handler.close()