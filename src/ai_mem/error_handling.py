"""🔐 Error handling with security-first approach.

Provides secure error handling that:
- Never leaks sensitive information
- Logs details for debugging
- Returns user-friendly messages
- Tracks error sources
"""

import logging
from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("ai_mem.errors")


class SecureErrorHandler:
    """Handle errors securely without leaking sensitive data."""
    
    # Map of error types to safe user messages
    ERROR_MESSAGES = {
        "validation": "Invalid input provided",
        "auth": "Authentication failed",
        "permission": "Permission denied",
        "not_found": "Resource not found",
        "conflict": "Resource already exists",
        "rate_limit": "Too many requests. Please try again later",
        "server": "Internal server error",
        "database": "Database error",
        "network": "Network error",
    }
    
    @staticmethod
    def log_error(error_type: str, message: str, exc: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log error securely without leaking sensitive data.
        
        Args:
            error_type: Type of error (validation, auth, etc.)
            message: Error message (details for logging)
            exc: Exception object if available
            context: Additional context (will be sanitized)
        """
        sanitized_context = {}
        if context:
            # Remove sensitive keys from context
            sensitive_keys = {"token", "password", "api_key", "secret", "auth"}
            for key, value in context.items():
                if not any(s in key.lower() for s in sensitive_keys):
                    sanitized_context[key] = value
        
        context_str = f" | context: {sanitized_context}" if sanitized_context else ""
        exc_str = f" | exception: {type(exc).__name__}: {str(exc)}" if exc else ""
        
        logger.error(f"[{error_type.upper()}] {message}{context_str}{exc_str}")
    
    @staticmethod
    def get_user_message(error_type: str) -> str:
        """Get user-friendly error message (never leaks details).
        
        Args:
            error_type: Type of error
            
        Returns:
            Safe message to show user
        """
        return SecureErrorHandler.ERROR_MESSAGES.get(error_type, "An error occurred")
    
    @staticmethod
    def response(
        error_type: str,
        status_code: int,
        message: Optional[str] = None,
        exc: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> JSONResponse:
        """Create a secure error response.
        
        Args:
            error_type: Type of error
            status_code: HTTP status code
            message: Detailed message for logging
            exc: Exception object
            context: Additional context
            
        Returns:
            JSONResponse with safe error message
        """
        # Log the error with full details
        if message:
            SecureErrorHandler.log_error(error_type, message, exc, context)
        
        # Return safe message to user
        user_message = SecureErrorHandler.get_user_message(error_type)
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error_type,
                "message": user_message,
                # Only include request ID for support (no details)
                "error_id": None,  # Could be generated for tracking
            }
        )
    
    @staticmethod
    def validation_error(message: str, exc: Optional[Exception] = None) -> JSONResponse:
        """Handle validation error."""
        return SecureErrorHandler.response(
            "validation",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            message,
            exc
        )
    
    @staticmethod
    def auth_error(message: str = "Authentication required") -> JSONResponse:
        """Handle authentication error."""
        return SecureErrorHandler.response(
            "auth",
            status.HTTP_401_UNAUTHORIZED,
            message
        )
    
    @staticmethod
    def permission_error(message: str = "Insufficient permissions") -> JSONResponse:
        """Handle permission error."""
        return SecureErrorHandler.response(
            "permission",
            status.HTTP_403_FORBIDDEN,
            message
        )
    
    @staticmethod
    def not_found_error(resource: str = "Resource") -> JSONResponse:
        """Handle not found error."""
        return SecureErrorHandler.response(
            "not_found",
            status.HTTP_404_NOT_FOUND,
            f"{resource} not found"
        )
    
    @staticmethod
    def conflict_error(message: str = "Resource conflict") -> JSONResponse:
        """Handle conflict error."""
        return SecureErrorHandler.response(
            "conflict",
            status.HTTP_409_CONFLICT,
            message
        )
    
    @staticmethod
    def rate_limit_error() -> JSONResponse:
        """Handle rate limit error."""
        return SecureErrorHandler.response(
            "rate_limit",
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Rate limit exceeded"
        )
    
    @staticmethod
    def server_error(message: str = "Internal server error", exc: Optional[Exception] = None) -> JSONResponse:
        """Handle server error."""
        logger.exception(f"Unhandled exception: {exc}")
        return SecureErrorHandler.response(
            "server",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            message,
            exc
        )
    
    @staticmethod
    def database_error(message: str, exc: Optional[Exception] = None) -> JSONResponse:
        """Handle database error."""
        return SecureErrorHandler.response(
            "database",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            message,
            exc
        )


class ErrorLogger:
    """Context manager for safe error logging."""
    
    def __init__(self, operation: str, sanitize_keys: Optional[list] = None):
        """Initialize error logger.
        
        Args:
            operation: Operation being performed (for logging)
            sanitize_keys: Keys to sanitize from context
        """
        self.operation = operation
        self.sanitize_keys = sanitize_keys or ["token", "password", "key", "secret"]
    
    def __enter__(self):
        """Enter context."""
        logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is not None:
            logger.error(
                f"Operation failed: {self.operation}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
            return False
        
        logger.debug(f"Operation completed: {self.operation}")
        return True
    
    def log_warning(self, message: str, context: Optional[Dict] = None):
        """Log warning with sanitization."""
        safe_context = self._sanitize(context or {})
        logger.warning(f"{self.operation}: {message} - {safe_context}")
    
    @staticmethod
    def _sanitize(context: Dict) -> Dict:
        """Remove sensitive keys from context."""
        sanitized = {}
        sensitive_keys = {"token", "password", "key", "secret", "auth"}
        for k, v in context.items():
            if not any(s in k.lower() for s in sensitive_keys):
                sanitized[k] = v
        return sanitized
