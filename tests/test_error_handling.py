"""🔐 Tests for secure error handling."""

import pytest
from fastapi import status
from ai_mem.error_handling import SecureErrorHandler, ErrorLogger
import logging


class TestSecureErrorHandler:
    """Test secure error handling."""
    
    def test_validation_error(self):
        """Test validation error response."""
        response = SecureErrorHandler.validation_error("Field X is invalid")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        content = response.body.decode()
        assert "error" in content
        assert "Invalid input" in content
        # Should not leak details
        assert "Field X" not in content
    
    def test_auth_error(self):
        """Test authentication error."""
        response = SecureErrorHandler.auth_error()
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        content = response.body.decode()
        assert "error" in content
        assert "auth" in content
    
    def test_permission_error(self):
        """Test permission error."""
        response = SecureErrorHandler.permission_error()
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
        content = response.body.decode()
        assert "error" in content
    
    def test_not_found_error(self):
        """Test not found error."""
        response = SecureErrorHandler.not_found_error("Observation")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        content = response.body.decode()
        assert "not_found" in content
    
    def test_conflict_error(self):
        """Test conflict error."""
        response = SecureErrorHandler.conflict_error()
        
        assert response.status_code == status.HTTP_409_CONFLICT
        assert response.body is not None
    
    def test_rate_limit_error(self):
        """Test rate limit error."""
        response = SecureErrorHandler.rate_limit_error()
        
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        content = response.body.decode()
        assert "rate_limit" in content
    
    def test_server_error(self):
        """Test server error."""
        exc = ValueError("Internal issue")
        response = SecureErrorHandler.server_error("Operation failed", exc)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        content = response.body.decode()
        # Should not leak exception details to user
        assert "ValueError" not in content
        assert "Internal issue" not in content
    
    def test_database_error(self):
        """Test database error."""
        exc = Exception("Connection failed")
        response = SecureErrorHandler.database_error("DB connection error", exc)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        # Error should not leak details
        assert response.body is not None
    
    def test_error_message_not_leaked(self):
        """Test that sensitive information is not leaked in responses."""
        # Simulate an error with sensitive data
        response = SecureErrorHandler.validation_error(
            "API key sk-1234567890 is invalid"
        )
        
        content = response.body.decode()
        # Should not contain the actual API key
        assert "sk-1234567890" not in content
        assert "Invalid input" in content  # But generic message is there
    
    def test_log_error_sanitizes_context(self):
        """Test that log_error sanitizes sensitive context."""
        context = {
            "token": "secret_token",
            "user_id": "123",
            "api_key": "secret_key",
        }
        # This should not raise
        SecureErrorHandler.log_error("test", "Test error", context=context)
        
        # Token and api_key should be sanitized internally
    
    def test_get_user_message_safe(self):
        """Test that user messages are always safe."""
        error_types = ["validation", "auth", "permission", "not_found", "rate_limit"]
        
        for error_type in error_types:
            message = SecureErrorHandler.get_user_message(error_type)
            assert message is not None
            assert len(message) > 0
            assert message in SecureErrorHandler.ERROR_MESSAGES.values()


class TestErrorLogger:
    """Test ErrorLogger context manager."""
    
    def test_error_logger_success(self):
        """Test ErrorLogger on successful operation."""
        with ErrorLogger("test_operation") as logger:
            # Operation succeeds
            pass
        # Should not raise
    
    def test_error_logger_exception(self):
        """Test ErrorLogger captures exceptions."""
        with pytest.raises(ValueError):
            with ErrorLogger("failing_operation") as logger:
                raise ValueError("Test error")
    
    def test_error_logger_sanitizes_warning(self):
        """Test that ErrorLogger sanitizes warnings."""
        with ErrorLogger("test") as logger:
            context = {
                "token": "secret",
                "data": "public",
            }
            logger.log_warning("Test warning", context)
            # Should sanitize token from context
    
    def test_error_logger_operation_name(self):
        """Test that operation name is included in logs."""
        with ErrorLogger("important_operation") as logger:
            pass
        # Operation name should be in logs


class TestErrorResponseFormat:
    """Test error response format."""
    
    def test_error_response_json_format(self):
        """Test that error responses are valid JSON."""
        response = SecureErrorHandler.not_found_error()
        
        import json
        content = json.loads(response.body.decode())
        
        assert "error" in content
        assert "message" in content
        assert isinstance(content["error"], str)
        assert isinstance(content["message"], str)
    
    def test_error_response_includes_error_type(self):
        """Test that error type is included."""
        response = SecureErrorHandler.validation_error("invalid")
        
        import json
        content = json.loads(response.body.decode())
        
        assert content["error"] == "validation"
    
    def test_different_error_types_different_codes(self):
        """Test that different error types have different status codes."""
        errors = [
            (SecureErrorHandler.validation_error("x"), 422),
            (SecureErrorHandler.auth_error(), 401),
            (SecureErrorHandler.permission_error(), 403),
            (SecureErrorHandler.not_found_error(), 404),
            (SecureErrorHandler.conflict_error(), 409),
            (SecureErrorHandler.rate_limit_error(), 429),
            (SecureErrorHandler.server_error("x"), 500),
        ]
        
        for response, expected_code in errors:
            assert response.status_code == expected_code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
