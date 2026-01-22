"""Path Validation - Improved path handling and security.

This module provides robust path validation to prevent:
- Incorrect tilde expansion
- Malformed paths
- Path traversal attacks
- Accidental file creation
- URL/path confusion

Configuration:
    AI_MEM_STRICT_PATH_VALIDATION=true
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

from .logging_config import get_logger

logger = get_logger("path_validation")

# Configuration
STRICT_VALIDATION = os.environ.get("AI_MEM_STRICT_PATH_VALIDATION", "true").lower() in ("true", "1", "yes")

# Patterns for detection
URL_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.-]*://')
WINDOWS_PATH_PATTERN = re.compile(r'^[A-Za-z]:[/\\]')
MALFORMED_TILDE_PATTERN = re.compile(r'^~[^/\\]')
TRAVERSAL_PATTERN = re.compile(r'(^|[/\\])\.\.([/\\]|$)')


class PathValidationError(Exception):
    """Error raised when path validation fails."""

    def __init__(self, message: str, path: str, suggestion: Optional[str] = None):
        self.message = message
        self.path = path
        self.suggestion = suggestion
        super().__init__(f"{message}: {path}" + (f" (suggestion: {suggestion})" if suggestion else ""))


def is_url(path: str) -> bool:
    """Check if a string is a URL.

    Args:
        path: String to check

    Returns:
        True if it's a URL
    """
    return bool(URL_PATTERN.match(path))


def is_valid_path(path: str) -> Tuple[bool, Optional[str]]:
    """Validate a file path.

    Args:
        path: Path to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path:
        return False, "Empty path"

    # Check if it's a URL (shouldn't be treated as path)
    if is_url(path):
        return False, "Path appears to be a URL"

    # Check for malformed tilde
    if MALFORMED_TILDE_PATTERN.match(path):
        return False, "Malformed tilde expansion (use ~/path not ~path)"

    # Check for path traversal
    if TRAVERSAL_PATTERN.search(path):
        return False, "Path contains directory traversal (..)"

    # Check for null bytes
    if '\x00' in path:
        return False, "Path contains null bytes"

    # Check for control characters
    if any(ord(c) < 32 for c in path if c != '\t'):
        return False, "Path contains control characters"

    return True, None


def expand_path(path: str, strict: bool = None) -> str:
    """Safely expand a path with tilde and environment variables.

    Args:
        path: Path to expand
        strict: Override strict validation setting

    Returns:
        Expanded path

    Raises:
        PathValidationError: If path is invalid
    """
    if strict is None:
        strict = STRICT_VALIDATION

    # Validate first
    is_valid, error = is_valid_path(path)
    if not is_valid:
        if strict:
            raise PathValidationError(error, path)
        else:
            logger.warning(f"Path validation warning: {error} for path: {path}")

    # Expand tilde
    if path.startswith("~"):
        path = os.path.expanduser(path)

    # Expand environment variables
    path = os.path.expandvars(path)

    return path


def normalize_path(path: str, strict: bool = None) -> str:
    """Normalize a path (expand, resolve, make absolute).

    Args:
        path: Path to normalize
        strict: Override strict validation setting

    Returns:
        Normalized absolute path

    Raises:
        PathValidationError: If path is invalid
    """
    expanded = expand_path(path, strict=strict)
    return str(Path(expanded).resolve())


def safe_join(base: Union[str, Path], *parts: str, strict: bool = None) -> str:
    """Safely join path components.

    Prevents path traversal by ensuring result stays within base.

    Args:
        base: Base directory
        *parts: Path components to join
        strict: Override strict validation setting

    Returns:
        Joined path

    Raises:
        PathValidationError: If result would escape base directory
    """
    if strict is None:
        strict = STRICT_VALIDATION

    base_path = Path(base).resolve()
    joined = base_path.joinpath(*parts).resolve()

    # Ensure result is within base
    try:
        joined.relative_to(base_path)
    except ValueError:
        if strict:
            raise PathValidationError(
                "Path would escape base directory",
                str(joined),
                suggestion=f"Ensure path stays within {base_path}",
            )
        logger.warning(f"Path traversal detected: {joined} escapes {base_path}")

    return str(joined)


def validate_file_path(
    path: str,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    strict: bool = None,
) -> str:
    """Validate and normalize a file path.

    Args:
        path: Path to validate
        must_exist: Path must exist
        must_be_file: Path must be a file
        must_be_dir: Path must be a directory
        strict: Override strict validation setting

    Returns:
        Normalized path

    Raises:
        PathValidationError: If validation fails
    """
    if strict is None:
        strict = STRICT_VALIDATION

    # Normalize
    normalized = normalize_path(path, strict=strict)
    p = Path(normalized)

    # Check existence
    if must_exist and not p.exists():
        raise PathValidationError("Path does not exist", path)

    # Check type
    if must_be_file and p.exists() and not p.is_file():
        raise PathValidationError("Path is not a file", path)

    if must_be_dir and p.exists() and not p.is_dir():
        raise PathValidationError("Path is not a directory", path)

    return normalized


def suggest_correction(path: str) -> Optional[str]:
    """Suggest a corrected path.

    Args:
        path: Potentially malformed path

    Returns:
        Suggested correction or None
    """
    # Fix malformed tilde
    if MALFORMED_TILDE_PATTERN.match(path):
        return "~/" + path[1:]

    # Fix double slashes
    if "//" in path:
        return re.sub(r'/+', '/', path)

    # Fix backslashes on Unix
    if os.name != 'nt' and '\\' in path:
        return path.replace('\\', '/')

    return None


def extract_path_from_url(url: str) -> Optional[str]:
    """Extract the path component from a URL.

    Args:
        url: URL string

    Returns:
        Path component or None
    """
    if not is_url(url):
        return None

    try:
        parsed = urlparse(url)
        return parsed.path or None
    except Exception:
        return None


def is_safe_filename(filename: str) -> Tuple[bool, Optional[str]]:
    """Check if a filename is safe.

    Args:
        filename: Filename to check

    Returns:
        Tuple of (is_safe, error_message)
    """
    if not filename:
        return False, "Empty filename"

    # Check for path separators
    if '/' in filename or '\\' in filename:
        return False, "Filename contains path separators"

    # Check for reserved names (Windows)
    reserved = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
                'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
    if filename.upper().split('.')[0] in reserved:
        return False, "Reserved filename"

    # Check for dots
    if filename in ('.', '..'):
        return False, "Invalid filename"

    # Check for hidden files (might be intentional)
    if filename.startswith('.'):
        # This is often intentional, just log
        logger.debug(f"Hidden file: {filename}")

    return True, None


class PathValidator:
    """Configurable path validator.

    Usage:
        validator = PathValidator(strict=True)
        path = validator.validate("/path/to/file")
    """

    def __init__(
        self,
        strict: bool = None,
        base_dir: Optional[str] = None,
        allow_urls: bool = False,
    ):
        """Initialize validator.

        Args:
            strict: Enable strict validation
            base_dir: Restrict paths to this directory
            allow_urls: Allow URLs (don't treat as error)
        """
        self.strict = strict if strict is not None else STRICT_VALIDATION
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self.allow_urls = allow_urls

    def validate(self, path: str) -> str:
        """Validate and normalize a path.

        Args:
            path: Path to validate

        Returns:
            Normalized path

        Raises:
            PathValidationError: If validation fails
        """
        # Check for URL
        if is_url(path):
            if not self.allow_urls:
                raise PathValidationError("URLs not allowed", path)
            return path

        # Validate
        normalized = normalize_path(path, strict=self.strict)

        # Check base directory restriction
        if self.base_dir:
            try:
                Path(normalized).relative_to(self.base_dir)
            except ValueError:
                raise PathValidationError(
                    f"Path must be within {self.base_dir}",
                    path,
                )

        return normalized

    def is_valid(self, path: str) -> bool:
        """Check if path is valid without raising.

        Args:
            path: Path to check

        Returns:
            True if valid
        """
        try:
            self.validate(path)
            return True
        except PathValidationError:
            return False
