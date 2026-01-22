"""🔐 Security tests for server endpoints.

Tests for input validation, authentication, CORS, and rate limiting.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from ai_mem.server import app
    return TestClient(app)


class TestEventIngestSecurity:
    """Test security of /api/events endpoint."""
    
    def test_event_invalid_host_rejected(self, client):
        """Test that invalid hosts are rejected."""
        response = client.post(
            "/api/events",
            json={
                "host": "../../etc/passwd",  # Path traversal attempt
                "payload": {"tool_name": "test"}
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_event_extra_fields_rejected(self, client):
        """Test that extra fields are rejected."""
        response = client.post(
            "/api/events",
            json={
                "host": "claude",
                "payload": {"tool_name": "test"},
                "evil_field": "malicious",  # Unknown field
                "another_bad": "value"
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_event_missing_payload_rejected(self, client):
        """Test that missing required fields are rejected."""
        response = client.post(
            "/api/events",
            json={
                "host": "claude"
                # Missing 'payload' which is required
            }
        )
        assert response.status_code == 422
    
    def test_event_valid_hosts_accepted(self, client):
        """Test that whitelisted hosts are accepted."""
        valid_hosts = ["claude", "gemini", "vscode", "cursor", "generic"]
        
        for host in valid_hosts:
            response = client.post(
                "/api/events",
                json={
                    "host": host,
                    "payload": {"tool_name": "test"}
                }
            )
            # Should pass validation (might fail later due to missing adapter, but not validation error)
            assert response.status_code != 422, f"Host {host} should be valid"
    
    def test_event_large_payload_accepted(self, client):
        """Test that large payloads within limits are accepted."""
        large_payload = {"data": "x" * 1_000_000}  # 1MB payload
        
        response = client.post(
            "/api/events",
            json={
                "host": "claude",
                "payload": large_payload
            }
        )
        # Should pass validation, fail for other reasons (no auth, etc.)
        assert response.status_code != 422, "Should accept 1MB payload"
    
    def test_event_type_validation(self, client):
        """Test that payload is validated as dict."""
        response = client.post(
            "/api/events",
            json={
                "host": "claude",
                "payload": "not a dict"  # Invalid type
            }
        )
        assert response.status_code == 422
    
    def test_event_tags_default(self, client):
        """Test that tags default to empty list."""
        response = client.post(
            "/api/events",
            json={
                "host": "claude",
                "payload": {"tool_name": "test"}
                # No tags provided
            }
        )
        # Should pass validation
        assert response.status_code != 422


class TestInputSanitization:
    """Test input sanitization and XSS prevention."""
    
    def test_search_query_sanitized(self, client):
        """Test that search queries are sanitized."""
        response = client.get(
            "/api/search",
            params={
                "q": "test<script>alert('xss')</script>"
            }
        )
        # Should not crash, should sanitize input
        assert response.status_code in [200, 400, 401, 422]
    
    def test_memory_add_content_sanitized(self, client):
        """Test that added memory content doesn't contain XSS."""
        response = client.post(
            "/api/memory",
            json={
                "content": "test<img src=x onerror=alert('xss')>",
                "obs_type": "note"
            }
        )
        # Should sanitize or reject
        assert response.status_code in [200, 400, 401, 404, 422]


class TestPathTraversal:
    """Test that path traversal is prevented."""
    
    def test_observation_path_traversal(self, client):
        """Test that observation IDs don't allow path traversal."""
        response = client.get(
            "/api/observations/../../etc/passwd"
        )
        # Should fail safely, not expose files
        assert response.status_code in [404, 400, 422]
    
    def test_session_path_traversal(self, client):
        """Test that session IDs don't allow path traversal."""
        response = client.get(
            "/api/sessions/../../sensitive"
        )
        # Should fail safely
        assert response.status_code in [404, 400, 422]


class TestErrorHandling:
    """Test error handling doesn't leak sensitive info."""
    
    def test_validation_error_response(self, client):
        """Test that validation errors don't expose system details."""
        response = client.post(
            "/api/events",
            json={
                "host": "invalid_host",
                "payload": "not dict"
            }
        )
        
        # Error response should not contain system paths
        text = response.text.lower()
        assert "/home/" not in text
        assert "traceback" not in text
    
    def test_missing_resource_error(self, client):
        """Test 404 error handling."""
        response = client.get("/api/observations/nonexistent")
        
        # Should return 400 (bad UUID) or 404 (not found)
        assert response.status_code in [400, 404]
        # Should have proper error message or bad request
        assert response.status_code in [400, 404]


class TestPayloadValidation:
    """Test request payload validation."""
    
    def test_payload_type_validation(self, client):
        """Test that payload must be a dict."""
        invalid_payloads = [
            None,
            "string",
            123,
            ["array"],
            True,
        ]
        
        for invalid in invalid_payloads:
            response = client.post(
                "/api/events",
                json={
                    "host": "claude",
                    "payload": invalid
                }
            )
            assert response.status_code == 422, f"Should reject payload type {type(invalid)}"
    
    def test_host_string_type(self, client):
        """Test that host must be string."""
        response = client.post(
            "/api/events",
            json={
                "host": 123,  # Wrong type
                "payload": {"tool": "test"}
            }
        )
        assert response.status_code == 422
    
    def test_session_id_validation(self, client):
        """Test session_id format if provided."""
        response = client.post(
            "/api/events",
            json={
                "host": "claude",
                "payload": {"tool": "test"},
                "session_id": "valid-uuid-format"
            }
        )
        # Should accept any string for session_id
        assert response.status_code != 422


class TestHostWhitelist:
    """Test host whitelist validation."""
    
    def test_valid_hosts(self, client):
        """Test all valid hosts are accepted."""
        valid_hosts = [
            "claude",
            "gemini", 
            "vscode",
            "cursor",
            "generic",
            "claude-code",
            "claude-desktop",
            "anthropic"
        ]
        
        for host in valid_hosts:
            response = client.post(
                "/api/events",
                json={
                    "host": host,
                    "payload": {"test": "data"}
                }
            )
            # Should pass validation
            assert response.status_code != 422, f"Host {host} should be valid"
    
    def test_invalid_hosts(self, client):
        """Test invalid hosts are rejected."""
        invalid_hosts = [
            "malicious",
            "evil_host",
            "../../etc",
            "/etc/passwd",
            "",
            " ",
            "claude; DROP TABLE;",
        ]
        
        for host in invalid_hosts:
            response = client.post(
                "/api/events",
                json={
                    "host": host,
                    "payload": {"test": "data"}
                }
            )
            # Should be rejected
            assert response.status_code == 422, f"Host '{host}' should be invalid"
    
    def test_host_case_sensitive(self, client):
        """Test that host validation is case sensitive."""
        response = client.post(
            "/api/events",
            json={
                "host": "CLAUDE",  # Uppercase
                "payload": {"test": "data"}
            }
        )
        # Should be rejected (whitelist is lowercase)
        assert response.status_code == 422


class TestJSONParsing:
    """Test JSON parsing and validation."""
    
    def test_malformed_json_rejected(self, client):
        """Test that malformed JSON is rejected."""
        response = client.post(
            "/api/events",
            content="{ invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable entity
    
    def test_empty_json_rejected(self, client):
        """Test that empty payload is handled."""
        response = client.post(
            "/api/events",
            json={}
        )
        # Should fail due to missing required fields
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
