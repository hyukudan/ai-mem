"""
Authentication service for ai-mem.

Provides JWT-based authentication, password hashing, and user management.
Self-hosted model with admin-controlled user creation.

Default admin credentials on first run:
- Email: admin@local
- Password: changeme
- MUST be changed on first login (must_change_password flag)
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..db import DatabaseManager

# Optional dependencies - graceful degradation
try:
    from jose import jwt, JWTError
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False
    JWTError = Exception

# Try to use bcrypt directly (more reliable than passlib with newer bcrypt versions)
try:
    import bcrypt as _bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False
    _bcrypt = None


from ..logging_config import get_logger
from ..db import DEFAULT_ADMIN_EMAIL, DEFAULT_ADMIN_PASSWORD

logger = get_logger("auth")

# Token configuration
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
ALGORITHM = "HS256"


class TokenPair(BaseModel):
    """JWT token pair response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class AuthResult(BaseModel):
    """Authentication result."""
    success: bool
    user_id: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    role: Optional[str] = None
    must_change_password: bool = False
    tokens: Optional[TokenPair] = None
    error: Optional[str] = None


class AuthService:
    """
    Authentication service for user management and JWT tokens.

    Features:
    - Password hashing with bcrypt
    - JWT access and refresh tokens
    - Admin-controlled user creation (no public registration)
    - Force password change on first login
    - OAuth support (optional, admin-configurable)
    """

    def __init__(
        self,
        db: "DatabaseManager",
        secret_key: Optional[str] = None,
        access_token_expire_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_token_expire_days: int = REFRESH_TOKEN_EXPIRE_DAYS,
    ):
        """
        Initialize auth service.

        Args:
            db: Database manager instance
            secret_key: JWT signing key (generated if not provided)
            access_token_expire_minutes: Access token TTL
            refresh_token_expire_days: Refresh token TTL
        """
        self.db = db
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.access_token_expire = access_token_expire_minutes
        self.refresh_token_expire = refresh_token_expire_days

        if not HAS_JOSE:
            logger.warning(
                "python-jose not installed. JWT tokens will not work. "
                "Install with: pip install python-jose[cryptography]"
            )
        if not HAS_BCRYPT:
            logger.warning(
                "bcrypt not installed. Password hashing will not work. "
                "Install with: pip install bcrypt"
            )

    # ==================== Password Hashing ====================

    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password

        Raises:
            RuntimeError: If bcrypt is not available
        """
        if HAS_BCRYPT and _bcrypt:
            # Use bcrypt directly for better compatibility
            password_bytes = password.encode('utf-8')
            salt = _bcrypt.gensalt(rounds=12)
            hashed = _bcrypt.hashpw(password_bytes, salt)
            return hashed.decode('utf-8')
        raise RuntimeError(
            "bcrypt not available. Install with: pip install bcrypt"
        )

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password
            hashed_password: Stored hash

        Returns:
            True if password matches
        """
        if not hashed_password:
            return False

        if HAS_BCRYPT and _bcrypt:
            try:
                password_bytes = plain_password.encode('utf-8')
                hashed_bytes = hashed_password.encode('utf-8')
                return _bcrypt.checkpw(password_bytes, hashed_bytes)
            except Exception:
                return False

        return False

    # ==================== JWT Tokens ====================

    def create_access_token(
        self,
        user_id: str,
        email: str,
        role: str,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Create a JWT access token.

        Args:
            user_id: User ID to encode
            email: User email
            role: User role
            expires_delta: Custom expiration time

        Returns:
            JWT token string
        """
        if not HAS_JOSE:
            raise RuntimeError("python-jose is required for JWT tokens")

        if expires_delta is None:
            expires_delta = timedelta(minutes=self.access_token_expire)

        expire = datetime.now(timezone.utc) + expires_delta

        payload = {
            "sub": user_id,
            "email": email,
            "role": role,
            "type": "access",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }

        return jwt.encode(payload, self.secret_key, algorithm=ALGORITHM)

    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Create a JWT refresh token.

        Args:
            user_id: User ID to encode
            expires_delta: Custom expiration time

        Returns:
            JWT refresh token string
        """
        if not HAS_JOSE:
            raise RuntimeError("python-jose is required for JWT tokens")

        if expires_delta is None:
            expires_delta = timedelta(days=self.refresh_token_expire)

        expire = datetime.now(timezone.utc) + expires_delta

        payload = {
            "sub": user_id,
            "type": "refresh",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_urlsafe(16),  # Unique token ID
        }

        return jwt.encode(payload, self.secret_key, algorithm=ALGORITHM)

    def create_token_pair(
        self,
        user_id: str,
        email: str,
        role: str,
    ) -> TokenPair:
        """
        Create access and refresh token pair.

        Args:
            user_id: User ID
            email: User email
            role: User role

        Returns:
            TokenPair with both tokens
        """
        access_token = self.create_access_token(user_id, email, role)
        refresh_token = self.create_refresh_token(user_id)

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_expire * 60,
        )

    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            Tuple of (success, payload or None)
        """
        if not HAS_JOSE:
            return False, None

        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[ALGORITHM])
            return True, payload
        except JWTError as e:
            logger.debug(f"Token verification failed: {e}")
            return False, None

    def verify_access_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify an access token specifically.

        Args:
            token: JWT access token

        Returns:
            Tuple of (success, payload with user info)
        """
        success, payload = self.verify_token(token)
        if not success or not payload:
            return False, None

        if payload.get("type") != "access":
            return False, None

        return True, {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "role": payload.get("role"),
        }

    def verify_refresh_token(self, token: str) -> Tuple[bool, Optional[str]]:
        """
        Verify a refresh token and return user_id.

        Args:
            token: JWT refresh token

        Returns:
            Tuple of (success, user_id or None)
        """
        success, payload = self.verify_token(token)
        if not success or not payload:
            return False, None

        if payload.get("type") != "refresh":
            return False, None

        return True, payload.get("sub")

    # ==================== Authentication ====================

    async def login(
        self,
        email: str,
        password: str,
    ) -> AuthResult:
        """
        Authenticate user with email and password.

        Args:
            email: User email
            password: Plain text password

        Returns:
            AuthResult with tokens if successful
        """
        # Get user by email
        user = await self.db.get_user_by_email(email)

        if not user:
            logger.warning(f"Login failed - user not found: {email}")
            return AuthResult(success=False, error="Invalid credentials")

        if not user.get("is_active"):
            logger.warning(f"Login failed - user inactive: {email}")
            return AuthResult(success=False, error="Account is disabled")

        # Verify password
        if not self.verify_password(password, user.get("password_hash", "")):
            logger.warning(f"Login failed - invalid password: {email}")
            return AuthResult(success=False, error="Invalid credentials")

        # Update last login
        await self.db.update_user(user["id"], last_login=time.time())

        # Create tokens
        tokens = self.create_token_pair(
            user_id=user["id"],
            email=user["email"],
            role=user["role"],
        )

        logger.info(f"Login successful: {email}")

        return AuthResult(
            success=True,
            user_id=user["id"],
            email=user["email"],
            name=user.get("name"),
            role=user["role"],
            must_change_password=user.get("must_change_password", False),
            tokens=tokens,
        )

    async def refresh_tokens(self, refresh_token: str) -> AuthResult:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            AuthResult with new tokens if successful
        """
        success, user_id = self.verify_refresh_token(refresh_token)

        if not success or not user_id:
            return AuthResult(success=False, error="Invalid refresh token")

        # Get user
        user = await self.db.get_user_by_id(user_id)

        if not user:
            return AuthResult(success=False, error="User not found")

        if not user.get("is_active"):
            return AuthResult(success=False, error="Account is disabled")

        # Create new tokens
        tokens = self.create_token_pair(
            user_id=user["id"],
            email=user["email"],
            role=user["role"],
        )

        return AuthResult(
            success=True,
            user_id=user["id"],
            email=user["email"],
            name=user.get("name"),
            role=user["role"],
            must_change_password=user.get("must_change_password", False),
            tokens=tokens,
        )

    async def get_current_user(self, access_token: str) -> Optional[Dict[str, Any]]:
        """
        Get current user from access token.

        Args:
            access_token: Valid access token

        Returns:
            User dict or None
        """
        success, payload = self.verify_access_token(access_token)

        if not success or not payload:
            return None

        user = await self.db.get_user_by_id(payload["user_id"])

        if not user or not user.get("is_active"):
            return None

        # Don't return password hash
        user.pop("password_hash", None)
        return user

    # ==================== Password Management ====================

    async def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str,
    ) -> Tuple[bool, str]:
        """
        Change user password.

        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password

        Returns:
            Tuple of (success, message)
        """
        user = await self.db.get_user_by_id(user_id)

        if not user:
            return False, "User not found"

        # Verify old password
        if not self.verify_password(old_password, user.get("password_hash", "")):
            return False, "Current password is incorrect"

        # Validate new password
        if len(new_password) < 8:
            return False, "Password must be at least 8 characters"

        # Update password
        new_hash = self.hash_password(new_password)
        success = await self.db.update_user(
            user_id,
            password_hash=new_hash,
            must_change_password=False,  # Clear the flag
        )

        if success:
            logger.info(f"Password changed for user: {user['email']}")
            return True, "Password changed successfully"

        return False, "Failed to update password"

    async def reset_password(
        self,
        user_id: str,
        new_password: str,
        require_change: bool = True,
    ) -> Tuple[bool, str]:
        """
        Admin reset of user password.

        Args:
            user_id: User ID
            new_password: New password
            require_change: Require password change on next login

        Returns:
            Tuple of (success, message)
        """
        user = await self.db.get_user_by_id(user_id)

        if not user:
            return False, "User not found"

        # Update password
        new_hash = self.hash_password(new_password)
        success = await self.db.update_user(
            user_id,
            password_hash=new_hash,
            must_change_password=require_change,
        )

        if success:
            logger.info(f"Password reset for user: {user['email']} (require_change={require_change})")
            return True, "Password reset successfully"

        return False, "Failed to reset password"

    # ==================== Admin User Management ====================

    async def create_user(
        self,
        email: str,
        password: str,
        name: Optional[str] = None,
        role: str = "user",
        require_password_change: bool = False,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Create a new user (admin only).

        Args:
            email: User email
            password: Initial password
            name: Display name
            role: User role (admin/user)
            require_password_change: Force password change on first login

        Returns:
            Tuple of (success, message, user_id or None)
        """
        # Validate email
        if not email or "@" not in email:
            return False, "Invalid email address", None

        # Validate password
        if len(password) < 8:
            return False, "Password must be at least 8 characters", None

        # Hash password
        password_hash = self.hash_password(password)

        # Create user
        user_id = await self.db.create_user(
            email=email,
            password_hash=password_hash,
            name=name,
            role=role,
            must_change_password=require_password_change,
        )

        if user_id:
            logger.info(f"User created by admin: {email} (role={role})")
            return True, "User created successfully", user_id

        return False, "Email already exists", None

    async def delete_user(self, user_id: str, admin_id: str) -> Tuple[bool, str]:
        """
        Delete a user (admin only).

        Args:
            user_id: User to delete
            admin_id: Admin performing the action

        Returns:
            Tuple of (success, message)
        """
        # Prevent self-deletion
        if user_id == admin_id:
            return False, "Cannot delete your own account"

        user = await self.db.get_user_by_id(user_id)
        if not user:
            return False, "User not found"

        success = await self.db.delete_user(user_id)

        if success:
            logger.info(f"User deleted by admin: {user['email']}")
            return True, "User deleted successfully"

        return False, "Failed to delete user"

    async def list_users(
        self,
        role: Optional[str] = None,
        active_only: bool = True,
    ) -> list[Dict[str, Any]]:
        """
        List all users (admin only).

        Args:
            role: Filter by role
            active_only: Only active users

        Returns:
            List of user dicts (without password hashes)
        """
        users = await self.db.list_users(role=role, active_only=active_only)

        # Remove password hashes
        for user in users:
            user.pop("password_hash", None)

        return users

    async def update_user_role(
        self,
        user_id: str,
        new_role: str,
        admin_id: str,
    ) -> Tuple[bool, str]:
        """
        Change user role (admin only).

        Args:
            user_id: User to update
            new_role: New role (admin/user)
            admin_id: Admin performing the action

        Returns:
            Tuple of (success, message)
        """
        if new_role not in ("admin", "user"):
            return False, "Invalid role"

        # Prevent admin from demoting themselves
        if user_id == admin_id and new_role != "admin":
            return False, "Cannot remove your own admin privileges"

        user = await self.db.get_user_by_id(user_id)
        if not user:
            return False, "User not found"

        success = await self.db.update_user(user_id, role=new_role)

        if success:
            logger.info(f"User role changed: {user['email']} -> {new_role}")
            return True, f"User role changed to {new_role}"

        return False, "Failed to update role"

    async def toggle_user_active(
        self,
        user_id: str,
        admin_id: str,
    ) -> Tuple[bool, str]:
        """
        Enable/disable a user (admin only).

        Args:
            user_id: User to toggle
            admin_id: Admin performing the action

        Returns:
            Tuple of (success, message)
        """
        # Prevent self-disable
        if user_id == admin_id:
            return False, "Cannot disable your own account"

        user = await self.db.get_user_by_id(user_id)
        if not user:
            return False, "User not found"

        new_status = not user.get("is_active", True)
        success = await self.db.update_user(user_id, is_active=new_status)

        if success:
            status_str = "enabled" if new_status else "disabled"
            logger.info(f"User {status_str}: {user['email']}")
            return True, f"User {status_str}"

        return False, "Failed to update user"

    # ==================== OAuth Support ====================

    async def oauth_login(
        self,
        provider: str,
        oauth_id: str,
        email: str,
        name: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> AuthResult:
        """
        Login or register via OAuth provider.

        Args:
            provider: OAuth provider (google, github)
            oauth_id: OAuth user ID
            email: User email from OAuth
            name: Display name from OAuth
            avatar_url: Avatar URL from OAuth

        Returns:
            AuthResult with tokens
        """
        # Check for existing OAuth user
        user = await self.db.get_user_by_oauth(provider, oauth_id)

        if not user:
            # Check if email exists (link accounts)
            user = await self.db.get_user_by_email(email)

            if user:
                # Link OAuth to existing account
                await self.db.update_user(
                    user["id"],
                    oauth_provider=provider,
                    oauth_id=oauth_id,
                )
                logger.info(f"OAuth linked to existing account: {email} ({provider})")
            else:
                # Create new user via OAuth
                user_id = await self.db.create_user(
                    email=email,
                    name=name,
                    oauth_provider=provider,
                    oauth_id=oauth_id,
                    must_change_password=False,
                )

                if not user_id:
                    return AuthResult(success=False, error="Failed to create account")

                user = await self.db.get_user_by_id(user_id)
                logger.info(f"User created via OAuth: {email} ({provider})")

        if not user:
            return AuthResult(success=False, error="Authentication failed")

        if not user.get("is_active"):
            return AuthResult(success=False, error="Account is disabled")

        # Update last login and avatar
        await self.db.update_user(
            user["id"],
            last_login=time.time(),
            avatar_url=avatar_url or user.get("avatar_url"),
            name=name or user.get("name"),
        )

        # Create tokens
        tokens = self.create_token_pair(
            user_id=user["id"],
            email=user["email"],
            role=user["role"],
        )

        return AuthResult(
            success=True,
            user_id=user["id"],
            email=user["email"],
            name=user.get("name") or name,
            role=user["role"],
            must_change_password=False,  # OAuth users don't need to change password
            tokens=tokens,
        )

    # ==================== Initialization ====================

    async def ensure_default_admin(self) -> Optional[str]:
        """
        Ensure default admin exists on first run.

        Creates admin@local with 'changeme' password if no users exist.
        Password MUST be changed on first login.

        Returns:
            Admin user ID if created, None if users exist
        """
        password_hash = self.hash_password(DEFAULT_ADMIN_PASSWORD)
        user_id = await self.db.ensure_default_admin(password_hash)

        if user_id:
            logger.warning(
                "\n"
                "=" * 60 + "\n"
                "  DEFAULT ADMIN ACCOUNT CREATED\n"
                f"  Email: {DEFAULT_ADMIN_EMAIL}\n"
                f"  Password: {DEFAULT_ADMIN_PASSWORD}\n"
                "  \n"
                "  *** YOU MUST CHANGE THIS PASSWORD ON FIRST LOGIN ***\n"
                "=" * 60
            )

        return user_id
