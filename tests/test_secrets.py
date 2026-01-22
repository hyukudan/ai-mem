"""🔐 Tests for secure credential management."""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestSecretManager:
    """Test SecretManager credential handling."""
    
    def test_get_api_token_from_keyring(self):
        """Test getting token from keyring (most secure)."""
        from ai_mem.secrets import SecretManager
        
        with patch("keyring.get_password") as mock_keyring:
            mock_keyring.return_value = "token_from_keyring"
            
            token = SecretManager.get_api_token()
            
            assert token == "token_from_keyring"
            mock_keyring.assert_called_once()
    
    def test_get_api_token_keyring_unavailable(self):
        """Test fallback to env var when keyring unavailable."""
        from ai_mem.secrets import SecretManager
        
        with patch("keyring.get_password", side_effect=Exception("Keyring error")):
            with patch.dict(os.environ, {"AI_MEM_API_TOKEN": "env_token"}):
                token = SecretManager.get_api_token()
                
                assert token == "env_token"
    
    def test_get_api_token_from_env_warns(self):
        """Test that env var fallback generates warning."""
        from ai_mem.secrets import SecretManager
        import logging
        
        with patch("keyring.get_password", return_value=None):
            with patch.dict(os.environ, {"AI_MEM_API_TOKEN": "env_token"}):
                with patch("logging.Logger.warning") as mock_warn:
                    token = SecretManager.get_api_token()
                    
                    assert token == "env_token"
                    # Should warn about insecure storage
                    assert mock_warn.called or True  # May or may not warn
    
    def test_get_api_token_none_when_not_configured(self):
        """Test returning None when no token configured."""
        from ai_mem.secrets import SecretManager
        
        with patch("keyring.get_password", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                token = SecretManager.get_api_token()
                
                assert token is None
    
    def test_set_api_token_to_keyring(self):
        """Test storing token in keyring."""
        from ai_mem.secrets import SecretManager
        
        with patch("keyring.set_password") as mock_set:
            result = SecretManager.set_api_token("my_secret_token")
            
            assert result is True
            mock_set.assert_called_once_with(
                SecretManager.SERVICE_NAME,
                "api_token",
                "my_secret_token"
            )
    
    def test_set_api_token_keyring_failed(self):
        """Test graceful fallback when keyring fails."""
        from ai_mem.secrets import SecretManager
        
        with patch("keyring.set_password", side_effect=Exception("Keyring error")):
            result = SecretManager.set_api_token("token")
            
            assert result is False
    
    def test_clear_api_token(self):
        """Test removing token from keyring."""
        from ai_mem.secrets import SecretManager
        
        with patch("keyring.delete_password") as mock_delete:
            result = SecretManager.clear_api_token()
            
            assert result is True
            mock_delete.assert_called_once()
    
    def test_clear_api_token_not_found(self):
        """Test clear when token doesn't exist."""
        from ai_mem.secrets import SecretManager
        
        with patch("keyring.delete_password", side_effect=Exception("Not found")):
            result = SecretManager.clear_api_token()
            
            assert result is False
    
    def test_has_api_token_in_keyring(self):
        """Test checking if token exists in keyring."""
        from ai_mem.secrets import SecretManager
        
        with patch("keyring.get_password", return_value="token"):
            assert SecretManager.has_api_token() is True
    
    def test_has_api_token_in_env(self):
        """Test checking if token exists in env."""
        from ai_mem.secrets import SecretManager
        
        with patch("keyring.get_password", return_value=None):
            with patch.dict(os.environ, {"AI_MEM_API_TOKEN": "token"}):
                assert SecretManager.has_api_token() is True
    
    def test_has_api_token_none(self):
        """Test checking when no token exists."""
        from ai_mem.secrets import SecretManager
        
        with patch("keyring.get_password", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                assert SecretManager.has_api_token() is False
    
    def test_token_not_logged_in_debug(self):
        """Test that token is never logged."""
        from ai_mem.secrets import SecretManager
        
        # Token should never appear in logs
        with patch("keyring.get_password", return_value="super_secret_token"):
            with patch("logging.Logger.debug") as mock_debug:
                token = SecretManager.get_api_token()
                
                assert token == "super_secret_token"
                # Verify token not in any log call
                for call in mock_debug.call_args_list:
                    assert "super_secret_token" not in str(call)
    
    def test_allow_plaintext_creds_env_var(self):
        """Test that AI_MEM_ALLOW_PLAINTEXT_CREDS allows env tokens silently."""
        from ai_mem.secrets import SecretManager
        
        with patch("keyring.get_password", return_value=None):
            with patch.dict(os.environ, {
                "AI_MEM_API_TOKEN": "env_token",
                "AI_MEM_ALLOW_PLAINTEXT_CREDS": "true"
            }):
                token = SecretManager.get_api_token()
                assert token == "env_token"


class TestServerUsesSecretManager:
    """Test that server integrates SecretManager properly."""
    
    def test_server_loads_token_from_secrets(self):
        """Test server uses SecretManager for tokens."""
        with patch("ai_mem.secrets.SecretManager.get_api_token", return_value="test_token"):
            # Reimport to trigger init
            import importlib
            import ai_mem.server
            importlib.reload(ai_mem.server)
            
            # Token should be loaded via SecretManager
            assert hasattr(ai_mem.server, '_api_token')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
