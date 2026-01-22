"""🔐 Secure credential management using system keyring.

This module provides secure storage and retrieval of sensitive credentials
like API tokens and keys using the system's secure storage backend.
"""

import os
from typing import Optional
import logging

logger = logging.getLogger("ai_mem.secrets")


class SecretManager:
    """Manage credentials securely using system keyring."""
    
    SERVICE_NAME = "ai-mem"
    
    @staticmethod
    def get_api_token(fallback_env: str = "AI_MEM_API_TOKEN") -> Optional[str]:
        """Get API token from secure storage or environment.
        
        Priority:
        1. System keyring (most secure)
        2. Environment variable (fallback)
        
        Args:
            fallback_env: Environment variable name to check as fallback
            
        Returns:
            API token if found, None otherwise
        """
        try:
            import keyring
            
            # Try keyring first (macOS Keychain, Windows Credential Manager, Linux)
            token = keyring.get_password(SecretManager.SERVICE_NAME, "api_token")
            if token:
                logger.debug("API token loaded from system keyring")
                return token
        except Exception as e:
            logger.debug(f"Keyring unavailable: {e}")
        
        # Fallback: Check environment variable
        token = os.environ.get(fallback_env)
        if token:
            allow_plain = os.environ.get("AI_MEM_ALLOW_PLAINTEXT_CREDS") == "true"
            if not allow_plain:
                logger.warning(
                    f"⚠️  API token found in environment variable '{fallback_env}'. "
                    f"For better security, move it to system keyring: "
                    f"`ai-mem config set-token <your-token>`"
                )
        
        return token
    
    @staticmethod
    def set_api_token(token: str) -> bool:
        """Store API token securely in system keyring.
        
        Args:
            token: API token to store
            
        Returns:
            True if successful, False if fallback to environment needed
        """
        try:
            import keyring
            
            keyring.set_password(SecretManager.SERVICE_NAME, "api_token", token)
            logger.info("✓ API token saved securely to system keyring")
            return True
        except Exception as e:
            logger.error(f"Failed to store token in keyring: {e}")
            logger.warning(
                f"⚠️  Could not access system keyring. Set fallback: "
                f"export AI_MEM_API_TOKEN='{token}' or "
                f"export AI_MEM_ALLOW_PLAINTEXT_CREDS=true"
            )
            return False
    
    @staticmethod
    def clear_api_token() -> bool:
        """Remove API token from secure storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import keyring
            
            keyring.delete_password(SecretManager.SERVICE_NAME, "api_token")
            logger.info("✓ API token removed from system keyring")
            return True
        except Exception as e:
            logger.debug(f"Could not delete token from keyring: {e}")
            return False
    
    @staticmethod
    def has_api_token() -> bool:
        """Check if API token is configured.
        
        Returns:
            True if token exists in keyring or environment
        """
        try:
            import keyring
            token = keyring.get_password(SecretManager.SERVICE_NAME, "api_token")
            if token:
                return True
        except Exception:
            pass
        
        return bool(os.environ.get("AI_MEM_API_TOKEN"))
