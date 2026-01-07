"""Custom exceptions for ai-mem.

This module defines a hierarchy of exceptions for better error classification
and handling throughout the application.
"""


class AiMemError(Exception):
    """Base exception for all ai-mem errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(AiMemError):
    """Error in configuration settings."""
    pass


class MissingConfigError(ConfigurationError):
    """Required configuration value is missing."""

    def __init__(self, key: str, message: str = None):
        msg = message or f"Missing required configuration: {key}"
        super().__init__(msg, {"key": key})
        self.key = key


class InvalidConfigError(ConfigurationError):
    """Configuration value is invalid."""

    def __init__(self, key: str, value: str, message: str = None):
        msg = message or f"Invalid configuration value for {key}: {value}"
        super().__init__(msg, {"key": key, "value": value})
        self.key = key
        self.value = value


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(AiMemError):
    """Input validation failed."""

    def __init__(self, field: str, message: str, value: str = None):
        super().__init__(message, {"field": field, "value": value})
        self.field = field
        self.value = value


class InvalidUUIDError(ValidationError):
    """Invalid UUID format."""

    def __init__(self, value: str):
        super().__init__(
            field="id",
            message=f"Invalid UUID format: {value}",
            value=value,
        )


class InvalidDateError(ValidationError):
    """Invalid date format."""

    def __init__(self, value: str):
        super().__init__(
            field="date",
            message=f"Invalid date format: {value}. Expected YYYY-MM-DD or epoch timestamp.",
            value=value,
        )


# =============================================================================
# Network/API Errors
# =============================================================================


class NetworkError(AiMemError):
    """Network-related errors."""
    pass


class APIError(NetworkError):
    """Error from external API."""

    def __init__(self, provider: str, status_code: int = None, message: str = None):
        msg = message or f"API error from {provider}"
        if status_code:
            msg += f" (status: {status_code})"
        super().__init__(msg, {"provider": provider, "status_code": status_code})
        self.provider = provider
        self.status_code = status_code


class TimeoutError(NetworkError):
    """Request timed out."""

    def __init__(self, operation: str, timeout_seconds: float):
        msg = f"Operation '{operation}' timed out after {timeout_seconds}s"
        super().__init__(msg, {"operation": operation, "timeout": timeout_seconds})
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class ConnectionError(NetworkError):
    """Failed to connect."""

    def __init__(self, target: str, message: str = None):
        msg = message or f"Failed to connect to {target}"
        super().__init__(msg, {"target": target})
        self.target = target


# =============================================================================
# Database Errors
# =============================================================================


class DatabaseError(AiMemError):
    """Database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Failed to connect to database."""

    def __init__(self, db_path: str, message: str = None):
        msg = message or f"Failed to connect to database: {db_path}"
        super().__init__(msg, {"db_path": db_path})
        self.db_path = db_path


class DatabaseIntegrityError(DatabaseError):
    """Database integrity constraint violated."""

    def __init__(self, message: str, constraint: str = None):
        super().__init__(message, {"constraint": constraint})
        self.constraint = constraint


# =============================================================================
# Adapter Errors
# =============================================================================


class AdapterError(AiMemError):
    """Error in host adapter."""
    pass


class UnknownHostError(AdapterError):
    """Unknown host identifier."""

    def __init__(self, host: str):
        msg = f"Unknown host: {host}"
        super().__init__(msg, {"host": host})
        self.host = host


class PayloadParseError(AdapterError):
    """Failed to parse event payload."""

    def __init__(self, host: str, message: str = None):
        msg = message or f"Failed to parse payload for host: {host}"
        super().__init__(msg, {"host": host})
        self.host = host


# =============================================================================
# Vector Store Errors
# =============================================================================


class VectorStoreError(AiMemError):
    """Error in vector store operations."""
    pass


class EmbeddingError(VectorStoreError):
    """Error generating embeddings."""

    def __init__(self, message: str, provider: str = None):
        super().__init__(message, {"provider": provider})
        self.provider = provider


# =============================================================================
# Resource Errors
# =============================================================================


class ResourceNotFoundError(AiMemError):
    """Requested resource not found."""

    def __init__(self, resource_type: str, resource_id: str):
        msg = f"{resource_type} not found: {resource_id}"
        super().__init__(msg, {"type": resource_type, "id": resource_id})
        self.resource_type = resource_type
        self.resource_id = resource_id


class ObservationNotFoundError(ResourceNotFoundError):
    """Observation not found."""

    def __init__(self, obs_id: str):
        super().__init__("Observation", obs_id)


class SessionNotFoundError(ResourceNotFoundError):
    """Session not found."""

    def __init__(self, session_id: str):
        super().__init__("Session", session_id)


class ProjectNotFoundError(ResourceNotFoundError):
    """Project not found."""

    def __init__(self, project: str):
        super().__init__("Project", project)


# =============================================================================
# Authentication/Authorization Errors
# =============================================================================


class AuthError(AiMemError):
    """Authentication/authorization errors."""
    pass


class InvalidTokenError(AuthError):
    """Invalid API token."""

    def __init__(self):
        super().__init__("Invalid or missing API token")


class UnauthorizedError(AuthError):
    """Operation not authorized."""

    def __init__(self, operation: str = None):
        msg = "Unauthorized"
        if operation:
            msg += f": {operation}"
        super().__init__(msg, {"operation": operation})
        self.operation = operation
