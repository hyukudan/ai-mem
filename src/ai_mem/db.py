import json
import aiosqlite
import sqlite3 # keep for Row type if needed, or row_factory logic
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable

from .exceptions import (
    DatabaseConnectionError,
    DatabaseIntegrityError,
)
from .logging_config import get_logger, log_duration
from .models import Observation, Session, ObservationIndex, User, UserRole

logger = get_logger("db")

# Default admin credentials - MUST be changed on first login
DEFAULT_ADMIN_EMAIL = "admin@local"
DEFAULT_ADMIN_PASSWORD = "changeme"


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        logger.debug(f"Connecting to database: {self.db_path}")
        start = time.perf_counter()
        try:
            self.conn = await aiosqlite.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = aiosqlite.Row
            await self._configure()
            await self.create_tables()
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(f"Database connected in {duration_ms:.2f}ms: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {self.db_path}: {e}")
            raise DatabaseConnectionError(str(self.db_path), str(e)) from e

    async def close(self) -> None:
        if self.conn:
            logger.debug("Closing database connection")
            await self.conn.close()
            logger.info("Database connection closed")

    async def _configure(self) -> None:
        if not self.conn:
            return
        await self.conn.execute("PRAGMA foreign_keys = ON")
        await self.conn.execute("PRAGMA journal_mode = WAL")
        await self.conn.execute("PRAGMA busy_timeout = 5000")
        await self.conn.commit()

    async def create_tables(self) -> None:
        if not self.conn:
            return
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                goal TEXT,
                summary TEXT,
                start_time REAL NOT NULL,
                end_time REAL
            )
            """
        )
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                project TEXT NOT NULL,
                type TEXT NOT NULL,
                concept TEXT,
                title TEXT,
                content TEXT NOT NULL,
                summary TEXT,
                content_hash TEXT,
                created_at REAL NOT NULL,
                importance_score REAL DEFAULT 0.5,
                tags TEXT,
                metadata TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        await self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
                title,
                summary,
                content,
                tags,
                content='observations',
                content_rowid='rowid'
            )
            """
        )
        await self.conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
                INSERT INTO observations_fts(rowid, title, summary, content, tags)
                VALUES (new.rowid, new.title, new.summary, new.content, new.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations BEGIN
                INSERT INTO observations_fts(observations_fts, rowid, title, summary, content, tags)
                VALUES('delete', old.rowid, old.title, old.summary, old.content, old.tags);
                INSERT INTO observations_fts(rowid, title, summary, content, tags)
                VALUES (new.rowid, new.title, new.summary, new.content, new.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
                INSERT INTO observations_fts(observations_fts, rowid, title, summary, content, tags)
                VALUES('delete', old.rowid, old.title, old.summary, old.content, old.tags);
            END;
            """
        )
        # Indexes for optimization
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_project ON observations(project)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_type ON observations(type)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_concept ON observations(concept)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_created ON observations(created_at)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_hash ON observations(content_hash)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(session_id)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_project_created ON observations(project, created_at)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_time)"
        )
        # Additional composite indexes for common query patterns
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_project_type ON observations(project, type)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_session_created ON observations(session_id, created_at DESC)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_type_created ON observations(type, created_at DESC)"
        )
        await self.conn.commit()
        await self._ensure_columns()
        await self._ensure_asset_table()
        await self._ensure_event_idempotency_table()
        await self._ensure_entity_tables()
        await self._ensure_user_tables()
        await self._ensure_user_id_columns()

    def _tag_clause(
        self,
        tag_filters: Optional[List[str]],
        params: List[Any],
        column: str = "tags",
    ) -> Optional[str]:
        if not tag_filters:
            return None
        valid_tags = [str(t).strip() for t in tag_filters if str(t).strip()]
        if not valid_tags:
            return None

        if column in {"tags", "observations.tags"}:
            # Optimization: Use FTS index for faster tag filtering
            # We construct a MATCH query like: "tag1" OR "tag2"
            # We wrap tags in double quotes for phrase matching to be safe
            def quote_tag(tag: str) -> str:
                escaped = tag.replace('"', '""')
                return f'"{escaped}"'
            quoted_tags = [quote_tag(tag) for tag in valid_tags]
            match_query = " OR ".join(quoted_tags)
            params.append(match_query)
            return "observations.rowid IN (SELECT rowid FROM observations_fts WHERE tags MATCH ?)"

        # Fallback for non-FTS columns (requires explicit validation to avoid SQL injection)
        allowed_columns = {"project", "session_id", "type", "observations.project", "observations.session_id", "observations.type"}
        if column not in allowed_columns:
             # If it's not a known column, we don't trust it for direct inclusion
             return None

        clauses = []
        for tag in valid_tags:
            clauses.append(f"{column} LIKE ?")
            params.append(f'%"{tag}"%')
        
        if not clauses:
            return None
        return "(" + " OR ".join(clauses) + ")"

    async def _ensure_columns(self) -> None:
        if not self.conn:
            return
        async with self.conn.execute("PRAGMA table_info(observations)") as cursor:
            rows = await cursor.fetchall()
            existing = {row["name"] for row in rows}

        if "content_hash" not in existing:
            await self.conn.execute("ALTER TABLE observations ADD COLUMN content_hash TEXT")
            await self.conn.commit()
        if "diff" not in existing:
            await self.conn.execute("ALTER TABLE observations ADD COLUMN diff TEXT")
            await self.conn.commit()
        # Memory consolidation fields (Phase 3)
        if "superseded_by" not in existing:
            await self.conn.execute("ALTER TABLE observations ADD COLUMN superseded_by TEXT")
            await self.conn.commit()
        if "access_count" not in existing:
            await self.conn.execute("ALTER TABLE observations ADD COLUMN access_count INTEGER DEFAULT 0")
            await self.conn.commit()
        if "last_accessed_at" not in existing:
            await self.conn.execute("ALTER TABLE observations ADD COLUMN last_accessed_at REAL")
            await self.conn.commit()
        # Incremental indexing field (Phase 5)
        if "last_indexed_at" not in existing:
            await self.conn.execute("ALTER TABLE observations ADD COLUMN last_indexed_at REAL")
            await self.conn.commit()
        if "index_version" not in existing:
            await self.conn.execute("ALTER TABLE observations ADD COLUMN index_version INTEGER DEFAULT 0")
            await self.conn.commit()
        # Concept field for semantic categorization (gotcha, trade-off, pattern, etc.)
        if "concept" not in existing:
            await self.conn.execute("ALTER TABLE observations ADD COLUMN concept TEXT")
            await self.conn.commit()

    async def _ensure_asset_table(self) -> None:
        if not self.conn:
            return
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observation_assets (
                id TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL,
                type TEXT NOT NULL,
                name TEXT,
                path TEXT,
                content TEXT,
                metadata TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(observation_id) REFERENCES observations(id) ON DELETE CASCADE
            )
            """
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assets_observation ON observation_assets(observation_id)"
        )
        await self.conn.commit()

    async def _ensure_event_idempotency_table(self) -> None:
        """Create table for tracking processed event IDs (idempotency)."""
        if not self.conn:
            return
        # No FK constraint - we just track the mapping for idempotency
        # The observation might be deleted later, but we still want to
        # remember the event_id was processed to avoid re-ingestion
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_idempotency (
                event_id TEXT PRIMARY KEY,
                observation_id TEXT,
                host TEXT,
                processed_at REAL NOT NULL
            )
            """
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_idempotency_processed ON event_idempotency(processed_at)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_idempotency_host ON event_idempotency(host)"
        )
        await self.conn.commit()

    async def _ensure_entity_tables(self) -> None:
        """Create tables for entity graph (Phase 4)."""
        if not self.conn:
            return
        # Entities table
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                project TEXT,
                metadata TEXT,
                mention_count INTEGER DEFAULT 1,
                first_seen REAL,
                last_seen REAL,
                UNIQUE(name, entity_type, project)
            )
            """
        )
        # Entity relations table
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_relations (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                observation_id TEXT,
                confidence REAL DEFAULT 1.0,
                metadata TEXT,
                created_at REAL,
                FOREIGN KEY(source_entity_id) REFERENCES entities(id),
                FOREIGN KEY(target_entity_id) REFERENCES entities(id)
            )
            """
        )
        # Indexes for entity queries
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_project ON entities(project)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_relations_source ON entity_relations(source_entity_id)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_relations_target ON entity_relations(target_entity_id)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_relations_type ON entity_relations(relation_type)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_relations_obs ON entity_relations(observation_id)"
        )
        await self.conn.commit()

    async def _ensure_user_tables(self) -> None:
        """Create tables for user management."""
        if not self.conn:
            return
        # Users table
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                name TEXT,
                avatar_url TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                oauth_provider TEXT,
                oauth_id TEXT,
                created_at REAL NOT NULL,
                last_login REAL,
                is_active INTEGER DEFAULT 1,
                must_change_password INTEGER DEFAULT 0,
                settings TEXT
            )
            """
        )
        # Indexes for user queries
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_oauth ON users(oauth_provider, oauth_id)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)"
        )
        await self.conn.commit()
        logger.debug("User tables ensured")

    async def _ensure_user_id_columns(self) -> None:
        """Add user_id columns to sessions and observations for multi-user isolation."""
        if not self.conn:
            return

        # Check sessions table
        async with self.conn.execute("PRAGMA table_info(sessions)") as cursor:
            rows = await cursor.fetchall()
            session_columns = {row["name"] for row in rows}

        if "user_id" not in session_columns:
            await self.conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")
            await self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)"
            )
            logger.info("Added user_id column to sessions table")
            await self.conn.commit()

        # Check observations table
        async with self.conn.execute("PRAGMA table_info(observations)") as cursor:
            rows = await cursor.fetchall()
            obs_columns = {row["name"] for row in rows}

        if "user_id" not in obs_columns:
            await self.conn.execute("ALTER TABLE observations ADD COLUMN user_id TEXT")
            await self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_user ON observations(user_id)"
            )
            logger.info("Added user_id column to observations table")
            await self.conn.commit()

        # Check entities table
        async with self.conn.execute("PRAGMA table_info(entities)") as cursor:
            rows = await cursor.fetchall()
            entity_columns = {row["name"] for row in rows}

        if "user_id" not in entity_columns:
            await self.conn.execute("ALTER TABLE entities ADD COLUMN user_id TEXT")
            await self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id)"
            )
            logger.info("Added user_id column to entities table")
            await self.conn.commit()

    # ==================== User Management Methods ====================

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get a user by email address.

        Args:
            email: User email

        Returns:
            User dict or None
        """
        if not self.conn:
            return None
        async with self.conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.lower(),),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_user(row) if row else None

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user by ID.

        Args:
            user_id: User ID

        Returns:
            User dict or None
        """
        if not self.conn:
            return None
        async with self.conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_user(row) if row else None

    async def get_user_by_oauth(
        self, provider: str, oauth_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a user by OAuth provider and ID.

        Args:
            provider: OAuth provider (google, github, etc.)
            oauth_id: OAuth user ID

        Returns:
            User dict or None
        """
        if not self.conn:
            return None
        async with self.conn.execute(
            "SELECT * FROM users WHERE oauth_provider = ? AND oauth_id = ?",
            (provider, oauth_id),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_user(row) if row else None

    async def create_user(
        self,
        email: str,
        password_hash: Optional[str] = None,
        name: Optional[str] = None,
        role: str = "user",
        oauth_provider: Optional[str] = None,
        oauth_id: Optional[str] = None,
        must_change_password: bool = False,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Create a new user.

        Args:
            email: User email (unique)
            password_hash: Hashed password (None for OAuth-only)
            name: Display name
            role: User role (admin/user)
            oauth_provider: OAuth provider if using OAuth
            oauth_id: OAuth user ID
            must_change_password: Force password change on first login
            user_id: Optional custom user ID

        Returns:
            User ID if created, None if email exists
        """
        if not self.conn:
            return None

        user_id = user_id or str(uuid.uuid4())
        now = time.time()

        try:
            await self.conn.execute(
                """
                INSERT INTO users (
                    id, email, password_hash, name, role,
                    oauth_provider, oauth_id, created_at, is_active,
                    must_change_password, settings
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, '{}')
                """,
                (
                    user_id,
                    email.lower(),
                    password_hash,
                    name,
                    role,
                    oauth_provider,
                    oauth_id,
                    now,
                    1 if must_change_password else 0,
                ),
            )
            await self.conn.commit()
            logger.info(f"Created user: {email} (id={user_id}, role={role})")
            return user_id
        except sqlite3.IntegrityError:
            logger.warning(f"User creation failed - email exists: {email}")
            return None

    async def update_user(
        self,
        user_id: str,
        **kwargs,
    ) -> bool:
        """Update user fields.

        Args:
            user_id: User ID
            **kwargs: Fields to update (name, avatar_url, settings, etc.)

        Returns:
            True if updated
        """
        if not self.conn or not kwargs:
            return False

        # Allowed fields for update
        allowed_fields = {
            "name", "avatar_url", "settings", "is_active",
            "password_hash", "must_change_password", "last_login",
            "oauth_provider", "oauth_id", "role"
        }

        updates = []
        params = []
        for key, value in kwargs.items():
            if key not in allowed_fields:
                continue
            if key == "settings" and isinstance(value, dict):
                value = json.dumps(value)
            elif key in ("is_active", "must_change_password"):
                value = 1 if value else 0
            updates.append(f"{key} = ?")
            params.append(value)

        if not updates:
            return False

        params.append(user_id)
        sql = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"

        async with self.conn.execute(sql, params) as cursor:
            await self.conn.commit()
            return cursor.rowcount > 0

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user.

        Args:
            user_id: User ID

        Returns:
            True if deleted
        """
        if not self.conn:
            return False

        async with self.conn.execute(
            "DELETE FROM users WHERE id = ?",
            (user_id,),
        ) as cursor:
            await self.conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted user: {user_id}")

        return deleted

    async def list_users(
        self,
        role: Optional[str] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List all users.

        Args:
            role: Filter by role
            active_only: Only return active users
            limit: Maximum users to return

        Returns:
            List of user dicts
        """
        if not self.conn:
            return []

        conditions = []
        params: List[Any] = []

        if active_only:
            conditions.append("is_active = 1")
        if role:
            conditions.append("role = ?")
            params.append(role)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT * FROM users
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        return [self._row_to_user(row) for row in rows]

    async def get_user_count(self) -> int:
        """Get total user count.

        Returns:
            Number of users
        """
        if not self.conn:
            return 0

        async with self.conn.execute("SELECT COUNT(*) as count FROM users") as cursor:
            row = await cursor.fetchone()
            return row["count"] if row else 0

    async def ensure_default_admin(self, password_hash: str) -> Optional[str]:
        """Ensure a default admin user exists.

        Creates the default admin with must_change_password=True if no users exist.

        Args:
            password_hash: Hashed default password

        Returns:
            Admin user ID if created, None if users already exist
        """
        if not self.conn:
            return None

        # Check if any users exist
        user_count = await self.get_user_count()
        if user_count > 0:
            return None

        # Create default admin
        logger.warning(
            f"Creating default admin user: {DEFAULT_ADMIN_EMAIL} "
            "(password must be changed on first login)"
        )
        return await self.create_user(
            email=DEFAULT_ADMIN_EMAIL,
            password_hash=password_hash,
            name="Administrator",
            role="admin",
            must_change_password=True,
        )

    def _row_to_user(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to user dict."""
        settings = {}
        if row["settings"]:
            try:
                settings = json.loads(row["settings"])
            except json.JSONDecodeError:
                pass

        return {
            "id": row["id"],
            "email": row["email"],
            "password_hash": row["password_hash"],
            "name": row["name"],
            "avatar_url": row["avatar_url"],
            "role": row["role"],
            "oauth_provider": row["oauth_provider"],
            "oauth_id": row["oauth_id"],
            "created_at": row["created_at"],
            "last_login": row["last_login"],
            "is_active": bool(row["is_active"]),
            "must_change_password": bool(row["must_change_password"]),
            "settings": settings,
        }

    # ==================== Entity Graph Methods (Phase 4) ====================

    async def get_or_create_entity(
        self,
        name: str,
        entity_type: str,
        project: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> Optional[str]:
        """Get existing entity or create new one.

        Args:
            name: Entity name
            entity_type: Entity type
            project: Project identifier
            created_at: Timestamp

        Returns:
            Entity ID
        """
        if not self.conn:
            return None

        now = created_at or time.time()

        # Try to find existing entity
        async with self.conn.execute(
            """
            SELECT id FROM entities
            WHERE name = ? AND entity_type = ? AND (project = ? OR project IS NULL)
            """,
            (name, entity_type, project),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                # Update mention count and last_seen
                await self.conn.execute(
                    """
                    UPDATE entities
                    SET mention_count = mention_count + 1, last_seen = ?
                    WHERE id = ?
                    """,
                    (now, row["id"]),
                )
                await self.conn.commit()
                return row["id"]

        # Create new entity
        import uuid
        entity_id = str(uuid.uuid4())
        await self.conn.execute(
            """
            INSERT INTO entities (id, name, entity_type, project, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (entity_id, name, entity_type, project, now, now),
        )
        await self.conn.commit()
        return entity_id

    async def create_entity_relation(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        observation_id: Optional[str] = None,
        confidence: float = 1.0,
        created_at: Optional[float] = None,
    ) -> Optional[str]:
        """Create a relation between entities.

        Args:
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID
            relation_type: Type of relation
            observation_id: Source observation ID
            confidence: Confidence score
            created_at: Timestamp

        Returns:
            Relation ID
        """
        if not self.conn:
            return None

        import uuid
        relation_id = str(uuid.uuid4())
        now = created_at or time.time()

        await self.conn.execute(
            """
            INSERT OR IGNORE INTO entity_relations
            (id, source_entity_id, target_entity_id, relation_type, observation_id, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (relation_id, source_entity_id, target_entity_id, relation_type, observation_id, confidence, now),
        )
        await self.conn.commit()
        return relation_id

    async def get_related_entities(
        self,
        entity_id: str,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 1,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get entities related to a given entity.

        Args:
            entity_id: Starting entity ID
            relation_types: Filter by relation types
            max_depth: Maximum traversal depth
            limit: Maximum entities to return

        Returns:
            List of related entity dicts
        """
        if not self.conn:
            return []

        results: List[Dict[str, Any]] = []
        visited: set = {entity_id}
        current_ids = [entity_id]

        for depth in range(max_depth):
            if not current_ids:
                break

            placeholders = ",".join("?" * len(current_ids))
            params: List[Any] = list(current_ids)

            type_filter = ""
            if relation_types:
                type_placeholders = ",".join("?" * len(relation_types))
                type_filter = f"AND r.relation_type IN ({type_placeholders})"
                params.extend(relation_types)

            # Get outgoing relations
            sql = f"""
                SELECT DISTINCT e.id, e.name, e.entity_type, e.project,
                       e.mention_count, r.relation_type, {depth + 1} as depth
                FROM entity_relations r
                JOIN entities e ON r.target_entity_id = e.id
                WHERE r.source_entity_id IN ({placeholders}) {type_filter}
                UNION
                SELECT DISTINCT e.id, e.name, e.entity_type, e.project,
                       e.mention_count, r.relation_type, {depth + 1} as depth
                FROM entity_relations r
                JOIN entities e ON r.source_entity_id = e.id
                WHERE r.target_entity_id IN ({placeholders}) {type_filter}
            """
            params.extend(current_ids)
            if relation_types:
                params.extend(relation_types)

            next_ids = []
            async with self.conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    if row["id"] not in visited and len(results) < limit:
                        visited.add(row["id"])
                        next_ids.append(row["id"])
                        results.append({
                            "id": row["id"],
                            "name": row["name"],
                            "entity_type": row["entity_type"],
                            "project": row["project"],
                            "mention_count": row["mention_count"],
                            "relation_type": row["relation_type"],
                            "depth": row["depth"],
                        })

            current_ids = next_ids

        return results

    async def search_entities(
        self,
        query: str,
        project: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search for entities by name.

        Args:
            query: Search query
            project: Filter by project
            entity_types: Filter by types
            limit: Maximum entities to return

        Returns:
            List of matching entity dicts
        """
        if not self.conn:
            return []

        conditions = ["name LIKE ?"]
        params: List[Any] = [f"%{query}%"]

        if project:
            conditions.append("(project = ? OR project IS NULL)")
            params.append(project)

        if entity_types:
            placeholders = ",".join("?" * len(entity_types))
            conditions.append(f"entity_type IN ({placeholders})")
            params.extend(entity_types)

        sql = f"""
            SELECT id, name, entity_type, project, mention_count, first_seen, last_seen
            FROM entities
            WHERE {" AND ".join(conditions)}
            ORDER BY mention_count DESC, last_seen DESC
            LIMIT ?
        """
        params.append(limit)

        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "name": row["name"],
                "entity_type": row["entity_type"],
                "project": row["project"],
                "mention_count": row["mention_count"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
            }
            for row in rows
        ]

    async def get_entity_observations(
        self,
        entity_id: str,
        limit: int = 20,
    ) -> List[str]:
        """Get observation IDs that mention an entity.

        Args:
            entity_id: Entity ID
            limit: Maximum observations to return

        Returns:
            List of observation IDs
        """
        if not self.conn:
            return []

        sql = """
            SELECT DISTINCT observation_id
            FROM entity_relations
            WHERE (source_entity_id = ? OR target_entity_id = ?)
              AND observation_id IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
        """
        async with self.conn.execute(sql, (entity_id, entity_id, limit)) as cursor:
            rows = await cursor.fetchall()

        return [row["observation_id"] for row in rows]

    async def get_graph_stats(
        self,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics about the entity graph.

        Args:
            project: Filter by project

        Returns:
            Dict with graph statistics
        """
        if not self.conn:
            return {"entities": 0, "relations": 0, "by_type": {}}

        # Count entities
        if project:
            async with self.conn.execute(
                "SELECT COUNT(*) as count FROM entities WHERE project = ? OR project IS NULL",
                (project,)
            ) as cursor:
                row = await cursor.fetchone()
                entity_count = row["count"] if row else 0
        else:
            async with self.conn.execute("SELECT COUNT(*) as count FROM entities") as cursor:
                row = await cursor.fetchone()
                entity_count = row["count"] if row else 0

        # Count relations
        async with self.conn.execute("SELECT COUNT(*) as count FROM entity_relations") as cursor:
            row = await cursor.fetchone()
            relation_count = row["count"] if row else 0

        # Count by type
        async with self.conn.execute(
            "SELECT entity_type, COUNT(*) as count FROM entities GROUP BY entity_type"
        ) as cursor:
            rows = await cursor.fetchall()
            by_type = {row["entity_type"]: row["count"] for row in rows}

        # Count relations by type
        async with self.conn.execute(
            "SELECT relation_type, COUNT(*) as count FROM entity_relations GROUP BY relation_type"
        ) as cursor:
            rows = await cursor.fetchall()
            relations_by_type = {row["relation_type"]: row["count"] for row in rows}

        return {
            "entities": entity_count,
            "relations": relation_count,
            "entities_by_type": by_type,
            "relations_by_type": relations_by_type,
        }

    async def check_event_processed(self, event_id: str) -> Optional[str]:
        """Check if an event ID has already been processed.

        Args:
            event_id: The event ID to check

        Returns:
            The observation_id if already processed, None otherwise
        """
        if not self.conn or not event_id:
            return None
        async with self.conn.execute(
            "SELECT observation_id FROM event_idempotency WHERE event_id = ?",
            (event_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return row["observation_id"] if row else None

    async def record_event_processed(
        self,
        event_id: str,
        observation_id: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        """Record that an event ID has been processed.

        Args:
            event_id: The event ID that was processed
            observation_id: The resulting observation ID (if any)
            host: The host that generated the event
        """
        if not self.conn or not event_id:
            return
        await self.conn.execute(
            """
            INSERT OR IGNORE INTO event_idempotency (event_id, observation_id, host, processed_at)
            VALUES (?, ?, ?, ?)
            """,
            (event_id, observation_id, host, time.time()),
        )
        await self.conn.commit()

    async def cleanup_old_event_ids(self, max_age_days: int = 30) -> int:
        """Remove old event IDs to prevent table growth.

        Args:
            max_age_days: Remove events older than this many days

        Returns:
            Number of records deleted
        """
        if not self.conn:
            return 0
        cutoff = time.time() - (max_age_days * 86400)
        async with self.conn.execute(
            "DELETE FROM event_idempotency WHERE processed_at < ?",
            (cutoff,),
        ) as cursor:
            await self.conn.commit()
            return cursor.rowcount

    async def add_asset(
        self,
        observation_id: str,
        asset_type: str,
        name: Optional[str] = None,
        path: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        asset_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> None:
        if not self.conn:
            return
        if not asset_id:
            asset_id = str(uuid.uuid4())
        timestamp = created_at or time.time()
        metadata_value = json.dumps(metadata or {})
        await self.conn.execute(
            """
            INSERT OR REPLACE INTO observation_assets (
                id,
                observation_id,
                type,
                name,
                path,
                content,
                metadata,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                observation_id,
                asset_type,
                name,
                path,
                content,
                metadata_value,
                timestamp,
            ),
        )
        await self.conn.commit()

    async def get_assets_for_observations(self, observation_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        if not observation_ids:
            return {}
        if not self.conn:
            return {}
        
        # Handle large lists by chunking if necessary, but sqlite limit is typically high (999 default variables)
        placeholders = ",".join("?" for _ in observation_ids)
        async with self.conn.execute(
            f"SELECT * FROM observation_assets WHERE observation_id IN ({placeholders})",
            observation_ids,
        ) as cursor:
            rows = await cursor.fetchall()
            
        assets_map: Dict[str, List[Dict[str, Any]]] = {oid: [] for oid in observation_ids}
        for row in rows:
            obs_id = row["observation_id"]
            if obs_id in assets_map:
                assets_map[obs_id].append(
                    {
                        "id": row["id"],
                        "type": row["type"],
                        "name": row["name"],
                        "path": row["path"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"] or "{}"),
                        "created_at": row["created_at"],
                    }
                )
        return assets_map

    async def get_assets_for_observation(self, observation_id: str) -> List[Dict[str, Any]]:
        # Use the batch method for consistency
        assets_map = await self.get_assets_for_observations([observation_id])
        return assets_map.get(observation_id, [])

    def _row_to_asset(self, row: sqlite3.Row) -> Dict[str, Any]:
        metadata = json.loads(row["metadata"] or "{}")
        return {
            "id": row["id"],
            "observation_id": row["observation_id"],
            "type": row["type"],
            "name": row["name"],
            "path": row["path"],
            "content": row["content"],
            "metadata": metadata,
            "created_at": row["created_at"],
        }



    async def add_session(self, session: Session) -> None:
        if not self.conn:
            return
        await self.conn.execute(
            """
            INSERT OR REPLACE INTO sessions (id, project, goal, summary, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.project,
                session.goal,
                session.summary,
                session.start_time,
                session.end_time,
            ),
        )
        await self.conn.commit()

    async def add_observation(self, obs: Observation, user_id: Optional[str] = None) -> None:
        if not self.conn:
            logger.warning("Cannot add observation: database not connected")
            raise DatabaseConnectionError(str(self.db_path), "Database not connected")
        logger.debug(f"Adding observation: id={obs.id}, type={obs.type}, project={obs.project}, user_id={user_id}")
        start = time.perf_counter()
        try:
            await self.conn.execute(
                """
                INSERT OR REPLACE INTO observations (
                    id,
                    session_id,
                    project,
                    type,
                    concept,
                    title,
                    content,
                    summary,
                    content_hash,
                    created_at,
                    importance_score,
                    tags,
                    metadata,
                    diff,
                    user_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    obs.id,
                    obs.session_id,
                    obs.project,
                    obs.type,
                    obs.concept,
                    obs.title,
                    obs.content,
                    obs.summary,
                    obs.content_hash,
                    obs.created_at,
                    obs.importance_score,
                    json.dumps(obs.tags),
                    json.dumps(obs.metadata),
                    obs.diff,
                    user_id,
                ),
            )
            await self.conn.commit()
            duration_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"Observation added in {duration_ms:.2f}ms: {obs.id}")
        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error adding observation {obs.id}: {e}")
            raise DatabaseIntegrityError(str(e), constraint="observations") from e
        except Exception as e:
            logger.error(f"Error adding observation {obs.id}: {e}")
            raise

    async def find_observation_by_hash(
        self,
        content_hash: str,
        project: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.conn:
            return None
        async with self.conn.execute(
            """
            SELECT * FROM observations
            WHERE content_hash = ? AND project = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (content_hash, project),
        ) as cursor:
            row = await cursor.fetchone()
            
        if not row:
            return None
        # _row_to_observation expects assets or fetches them.
        # It's an internal helper. But _row_to_observation is sync if I didn't change it.
        # Wait, _row_to_observation calls self.get_assets_for_observation if assets is None.
        # But get_assets_for_observation is now ASYNC!
        # So _row_to_observation MUST be async OR I must fetch assets here.
        # I should fetch assets here.
        
        assets_map = await self.get_assets_for_observations([row["id"]])
        return self._row_to_observation(row, assets=assets_map.get(row["id"]))

    async def get_observation(self, obs_id: str) -> Optional[Dict[str, Any]]:
        if not self.conn:
            return None
        async with self.conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        assets_map = await self.get_assets_for_observations([obs_id])
        return self._row_to_observation(row, assets=assets_map.get(obs_id))

    async def get_observations(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        if not self.conn:
            return []
        placeholders = ",".join("?" for _ in ids)
        async with self.conn.execute(
            f"SELECT * FROM observations WHERE id IN ({placeholders})", ids
        ) as cursor:
            rows = await cursor.fetchall()
        
        # Batch fetch assets
        found_ids = [row["id"] for row in rows]
        assets_map = await self.get_assets_for_observations(found_ids)
        
        return [self._row_to_observation(row, assets=assets_map.get(row["id"])) for row in rows]

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.conn:
            return None
        async with self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_session(row)

    async def list_sessions(
        self,
        project: Optional[str] = None,
        active_only: bool = False,
        goal_query: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.conn:
            return []
        params: List[Any] = []
        sql = "SELECT * FROM sessions"
        conditions: List[str] = []
        if project:
            conditions.append("project = ?")
            params.append(project)
        if active_only:
            conditions.append("end_time IS NULL")
        if goal_query:
            conditions.append("LOWER(goal) LIKE ?")
            params.append(f"%{goal_query.lower()}%")
        if date_start is not None:
            conditions.append("start_time >= ?")
            params.append(date_start)
        if date_end is not None:
            conditions.append("start_time <= ?")
            params.append(date_end)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY start_time DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        
        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_session(row) for row in rows]

    async def list_observations(
        self,
        project: Optional[str] = None,
        limit: int = 50,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> List[ObservationIndex]:
        if not self.conn:
            return []
        params: List[Any] = []
        conditions: List[str] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if project:
            conditions.append("project = ?")
            params.append(project)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if obs_type:
            conditions.append("type = ?")
            params.append(obs_type)
        if date_start is not None:
            conditions.append("created_at >= ?")
            params.append(date_start)
        if date_end is not None:
            conditions.append("created_at <= ?")
            params.append(date_end)
        tag_clause = self._tag_clause(tag_filters, params)
        if tag_clause:
            conditions.append(tag_clause)
        
        sql = """
            SELECT id, summary, project, type, created_at
            FROM observations
        """
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        
        return [
            ObservationIndex(
                id=row["id"],
                summary=row["summary"] or "",
                project=row["project"],
                type=row["type"],
                created_at=row["created_at"],
                score=0.0,
            )
            for row in rows
        ]

    async def list_projects(self) -> List[str]:
        if not self.conn:
            return []
        async with self.conn.execute("SELECT DISTINCT project FROM observations ORDER BY project") as cursor:
            rows = await cursor.fetchall()
        return [row["project"] for row in rows]

    async def get_stats(
        self,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
        tag_limit: int = 10,
        day_limit: int = 14,
        type_tag_limit: int = 3,
    ) -> Dict[str, Any]:
        if not self.conn:
            return {}
        params: List[Any] = []
        conditions: List[str] = []
        if project:
            conditions.append("project = ?")
            params.append(project)
        if obs_type:
            conditions.append("type = ?")
            params.append(obs_type)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if date_start is not None:
            conditions.append("created_at >= ?")
            params.append(date_start)
        if date_end is not None:
            conditions.append("created_at <= ?")
            params.append(date_end)
        tag_clause = self._tag_clause(tag_filters, params)
        if tag_clause:
            conditions.append(tag_clause)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        async with self.conn.execute(f"SELECT COUNT(*) AS count FROM observations {where_clause}", params) as cursor:
            total_row = await cursor.fetchone()
        total = int(total_row["count"]) if total_row else 0

        end_ts = date_end
        if end_ts is None:
            async with self.conn.execute(
                f"SELECT MAX(created_at) AS max_time FROM observations {where_clause}",
                params,
            ) as cursor:
                max_row = await cursor.fetchone()
            if max_row and max_row["max_time"] is not None:
                end_ts = float(max_row["max_time"])

        async with self.conn.execute(
            f"""
            SELECT type, COUNT(*) AS count
            FROM observations
            {where_clause}
            GROUP BY type
            ORDER BY count DESC, type ASC
            """,
            params,
        ) as cursor:
            by_type = [
                {"type": row["type"], "count": int(row["count"])}
                for row in await cursor.fetchall()
            ]

        if project:
            by_project = [{"project": project, "count": total}] if total else []
        else:
            project_params: List[Any] = []
            project_conditions: List[str] = []
            if obs_type:
                project_conditions.append("type = ?")
                project_params.append(obs_type)
            if session_id:
                project_conditions.append("session_id = ?")
                project_params.append(session_id)
            if date_start is not None:
                project_conditions.append("created_at >= ?")
                project_params.append(date_start)
            if date_end is not None:
                project_conditions.append("created_at <= ?")
                project_params.append(date_end)
            tag_clause = self._tag_clause(tag_filters, project_params)
            if tag_clause:
                project_conditions.append(tag_clause)
            project_where = (
                f"WHERE {' AND '.join(project_conditions)}" if project_conditions else ""
            )
            async with self.conn.execute(
                f"""
                SELECT project, COUNT(*) AS count
                FROM observations
                {project_where}
                GROUP BY project
                ORDER BY count DESC, project ASC
                """,
                project_params,
            ) as cursor:
                by_project = [
                    {"project": row["project"], "count": int(row["count"])}
                    for row in await cursor.fetchall()
                ]

        by_day: List[Dict[str, Any]] = []
        recent_start: Optional[float] = None
        if day_limit > 0 and total and end_ts is not None:
            recent_start = end_ts - (day_limit * 86400)
            day_params = list(params)
            day_conditions = list(conditions)
            day_conditions.append("created_at >= ?")
            day_params.append(recent_start)
            day_conditions.append("created_at <= ?")
            day_params.append(end_ts)
            day_where = f"WHERE {' AND '.join(day_conditions)}" if day_conditions else ""
            day_params.append(day_limit)
            async with self.conn.execute(
                f"""
                SELECT strftime('%Y-%m-%d', created_at, 'unixepoch') AS day,
                       COUNT(*) AS count
                FROM observations
                {day_where}
                GROUP BY day
                ORDER BY day DESC
                LIMIT ?
                """,
                day_params,
            ) as cursor:
                by_day = [
                    {"day": row["day"], "count": int(row["count"])}
                    for row in await cursor.fetchall()
                ]
        recent_total = sum(item["count"] for item in by_day)
        previous_total = 0
        trend_delta = 0
        trend_pct: Optional[float] = None
        if day_limit > 0 and end_ts is not None and recent_start is not None:
            previous_start = recent_start - (day_limit * 86400)
            prev_params = list(params)
            prev_conditions = list(conditions)
            prev_conditions.append("created_at >= ?")
            prev_params.append(previous_start)
            prev_conditions.append("created_at < ?")
            prev_params.append(recent_start)
            prev_where = f"WHERE {' AND '.join(prev_conditions)}" if prev_conditions else ""
            async with self.conn.execute(
                f"SELECT COUNT(*) AS count FROM observations {prev_where}",
                prev_params,
            ) as cursor:
                prev_row = await cursor.fetchone()
            previous_total = int(prev_row["count"]) if prev_row else 0
            trend_delta = recent_total - previous_total
            if previous_total > 0:
                trend_pct = (trend_delta / previous_total) * 100.0

        top_tags: List[Dict[str, Any]] = []
        if tag_limit > 0 and total:
            async with self.conn.execute(
                f"SELECT tags FROM observations {where_clause}",
                params,
            ) as cursor:
                rows = await cursor.fetchall()
            counter: Counter[str] = Counter()
            for row in rows:
                raw = row["tags"]
                if not raw:
                    continue
                try:
                    tags = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(tags, list):
                    continue
                for tag in tags:
                    if not tag:
                        continue
                    counter[str(tag)] += 1
            top_tags = [
                {"tag": tag, "count": count}
                for tag, count in counter.most_common(tag_limit)
            ]

        top_tags_by_type: List[Dict[str, Any]] = []
        if not obs_type and type_tag_limit > 0 and total:
            async with self.conn.execute(f"SELECT type, tags FROM observations {where_clause}", params) as cursor:
                rows = await cursor.fetchall()
            per_type: Dict[str, Counter[str]] = {}
            for row in rows:
                obs_type_value = row["type"]
                raw = row["tags"]
                if not raw:
                    continue
                try:
                    tags = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(tags, list):
                    continue
                if obs_type_value not in per_type:
                    per_type[obs_type_value] = Counter()
                for tag in tags:
                    if not tag:
                        continue
                    per_type[obs_type_value][str(tag)] += 1
            top_tags_by_type = [
                {
                    "type": obs_type_value,
                    "tags": [
                        {"tag": tag, "count": count}
                        for tag, count in counter.most_common(type_tag_limit)
                    ],
                }
                for obs_type_value, counter in sorted(per_type.items(), key=lambda item: item[0])
            ]

        return {
            "total": total,
            "by_type": by_type,
            "by_project": by_project,
            "by_day": by_day,
            "top_tags": top_tags,
            "top_tags_by_type": top_tags_by_type,
            "recent_total": recent_total,
            "previous_total": previous_total,
            "trend_delta": trend_delta,
            "trend_pct": trend_pct,
            "day_limit": day_limit,
        }

    async def delete_observation(self, obs_id: str) -> int:
        if not self.conn:
            return 0
        async with self.conn.execute("DELETE FROM observations WHERE id = ?", (obs_id,)) as cursor:
            await self.conn.commit()
            return cursor.rowcount

    async def update_observation_tags(self, obs_id: str, tags: List[str]) -> int:
        if not self.conn:
            return 0
        async with self.conn.execute(
            "UPDATE observations SET tags = ? WHERE id = ?",
            (json.dumps(tags), obs_id),
        ) as cursor:
            await self.conn.commit()
            return cursor.rowcount

    async def delete_project(self, project: str) -> int:
        if not self.conn:
            return 0
        async with self.conn.execute("DELETE FROM observations WHERE project = ?", (project,)) as cursor:
            await self.conn.commit()
            return cursor.rowcount

    async def get_tag_counts(
        self,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        obs_type: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
        limit: Optional[int] = 50,
    ) -> List[Dict[str, Any]]:
        if not self.conn:
            return []
        params: List[Any] = []
        conditions: List[str] = []
        if project:
            conditions.append("project = ?")
            params.append(project)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if obs_type:
            conditions.append("type = ?")
            params.append(obs_type)
        if date_start is not None:
            conditions.append("created_at >= ?")
            params.append(date_start)
        if date_end is not None:
            conditions.append("created_at <= ?")
            params.append(date_end)
        tag_clause = self._tag_clause(tag_filters, params)
        if tag_clause:
            conditions.append(tag_clause)
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        async with self.conn.execute(f"SELECT tags FROM observations {where_clause}", params) as cursor:
            rows = await cursor.fetchall()
            
        counter = Counter()
        for row in rows:
            raw = row["tags"]
            try:
                tags = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                tags = []
            if not isinstance(tags, list):
                continue
            for tag in tags:
                if not tag:
                    continue
                counter[str(tag)] += 1
        items = [{"tag": tag, "count": count} for tag, count in counter.most_common()]
        if limit is not None and limit > 0:
            return items[:limit]
        return items

    async def replace_tag(
        self,
        old_tag: str,
        new_tag: Optional[str] = None,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        obs_type: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> int:
        if not self.conn:
            return 0
        value = str(old_tag or "").strip()
        if not value:
            return 0
        params: List[Any] = []
        conditions: List[str] = []
        if project:
            conditions.append("project = ?")
            params.append(project)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if obs_type:
            conditions.append("type = ?")
            params.append(obs_type)
        if date_start is not None:
            conditions.append("created_at >= ?")
            params.append(date_start)
        if date_end is not None:
            conditions.append("created_at <= ?")
            params.append(date_end)
        tag_clause = self._tag_clause([value], params)
        if tag_clause:
            conditions.append(tag_clause)
        extra_clause = self._tag_clause(tag_filters, params)
        if extra_clause:
            conditions.append(extra_clause)
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        async with self.conn.execute(f"SELECT id, tags FROM observations {where_clause}", params) as cursor:
            rows = await cursor.fetchall()
            
        updated = 0
        replacement = str(new_tag or "").strip() or None
        for row in rows:
            raw = row["tags"]
            try:
                tags = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                continue
            if not isinstance(tags, list):
                continue
            if value not in tags:
                continue
            if replacement:
                new_tags = [replacement if tag == value else tag for tag in tags]
            else:
                new_tags = [tag for tag in tags if tag != value]
            deduped = []
            seen = set()
            for tag in new_tags:
                tag_str = str(tag).strip()
                if not tag_str or tag_str in seen:
                    continue
                seen.add(tag_str)
                deduped.append(tag_str)
            if deduped == tags:
                continue
            await self.conn.execute(
                "UPDATE observations SET tags = ? WHERE id = ?",
                (json.dumps(deduped), row["id"]),
            )
            updated += 1
        if updated:
            await self.conn.commit()
        return updated

    async def add_tag(
        self,
        tag: str,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        obs_type: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> int:
        if not self.conn:
            return 0
        value = str(tag or "").strip()
        if not value:
            return 0
        params: List[Any] = []
        conditions: List[str] = []
        if project:
            conditions.append("project = ?")
            params.append(project)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if obs_type:
            conditions.append("type = ?")
            params.append(obs_type)
        if date_start is not None:
            conditions.append("created_at >= ?")
            params.append(date_start)
        if date_end is not None:
            conditions.append("created_at <= ?")
            params.append(date_end)
        tag_clause = self._tag_clause(tag_filters, params)
        if tag_clause:
            conditions.append(tag_clause)
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        async with self.conn.execute(f"SELECT id, tags FROM observations {where_clause}", params) as cursor:
            rows = await cursor.fetchall()
            
        updated = 0
        for row in rows:
            raw = row["tags"]
            try:
                tags = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                continue
            if not isinstance(tags, list):
                continue
            if value in tags:
                continue
            tags.append(value)
            deduped = []
            seen = set()
            for tag_item in tags:
                tag_str = str(tag_item).strip()
                if not tag_str or tag_str in seen:
                    continue
                seen.add(tag_str)
                deduped.append(tag_str)
            await self.conn.execute(
                "UPDATE observations SET tags = ? WHERE id = ?",
                (json.dumps(deduped), row["id"]),
            )
            updated += 1
        if updated:
            await self.conn.commit()
        return updated

    async def search_observations_fts(
        self,
        query: str,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
        limit: int = 20,
        user_id: Optional[str] = None,
    ) -> List[ObservationIndex]:
        if not self.conn:
            logger.warning("Cannot search: database not connected")
            return []
        logger.debug(f"FTS search: query='{query}', project={project}, user_id={user_id}, limit={limit}")
        start = time.perf_counter()
        conditions = []
        params: List[Any] = []

        if query.strip():
            # Escape double quotes and wrap in phrase markers to avoid FTS operator injection
            safe_query = '"' + query.replace('"', '""') + '"'
            conditions.append("observations_fts MATCH ?")
            params.append(safe_query)

        # User isolation - filter by user_id if provided
        if user_id:
            conditions.append("observations.user_id = ?")
            params.append(user_id)

        if project:
            conditions.append("observations.project = ?")
            params.append(project)

        if obs_type:
            conditions.append("observations.type = ?")
            params.append(obs_type)

        if session_id:
            conditions.append("observations.session_id = ?")
            params.append(session_id)

        if date_start is not None:
            conditions.append("observations.created_at >= ?")
            params.append(date_start)

        if date_end is not None:
            conditions.append("observations.created_at <= ?")
            params.append(date_end)

        tag_clause = self._tag_clause(tag_filters, params, column="tags")
        if tag_clause:
            conditions.append(tag_clause)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT
                observations.id,
                observations.summary,
                observations.project,
                observations.type,
                observations.created_at,
                bm25(observations_fts) AS rank
            FROM observations_fts
            JOIN observations ON observations.rowid = observations_fts.rowid
            {where_clause}
            ORDER BY rank ASC
            LIMIT ?
        """
        params.append(limit)
        
        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        duration_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"FTS search completed in {duration_ms:.2f}ms, found {len(rows)} results")

        return [
            ObservationIndex(
                id=row["id"],
                summary=row["summary"] or "",
                project=row["project"],
                type=row["type"],
                created_at=row["created_at"],
                score=1.0 / (1.0 + float(row["rank"] or 0.0)),
            )
            for row in rows
        ]

    async def get_recent_observations(
        self,
        project: Optional[str],
        limit: int,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> List[ObservationIndex]:
        if not self.conn:
            return []
        params: List[Any] = []
        sql = """
            SELECT id, summary, project, type, created_at
            FROM observations
        """
        conditions = []
        # User isolation - always filter by user_id first if provided
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if project:
            conditions.append("project = ?")
            params.append(project)
        if obs_type:
            conditions.append("type = ?")
            params.append(obs_type)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if date_start is not None:
            conditions.append("created_at >= ?")
            params.append(date_start)
        if date_end is not None:
            conditions.append("created_at <= ?")
            params.append(date_end)
        tag_clause = self._tag_clause(tag_filters, params)
        if tag_clause:
            conditions.append(tag_clause)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            
        return [
            ObservationIndex(
                id=row["id"],
                summary=row["summary"] or "",
                project=row["project"],
                type=row["type"],
                created_at=row["created_at"],
                score=0.0,
            )
            for row in rows
        ]

    async def get_observations_before(
        self,
        project: str,
        anchor_time: float,
        limit: int,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
    ) -> List[ObservationIndex]:
        if not self.conn:
            return []
        params: List[Any] = [project, anchor_time]
        conditions = ["project = ?", "created_at < ?"]
        if obs_type:
            conditions.append("type = ?")
            params.append(obs_type)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if date_start is not None:
            conditions.append("created_at >= ?")
            params.append(date_start)
        if date_end is not None:
            conditions.append("created_at <= ?")
            params.append(date_end)
        tag_clause = self._tag_clause(tag_filters, params)
        if tag_clause:
            conditions.append(tag_clause)
        sql = """
            SELECT id, summary, project, type, created_at
            FROM observations
            WHERE {conditions}
            ORDER BY created_at DESC
            LIMIT ?
        """.format(conditions=" AND ".join(conditions))
        params.append(limit)
        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [
            ObservationIndex(
                id=row["id"],
                summary=row["summary"] or "",
                project=row["project"],
                type=row["type"],
                created_at=row["created_at"],
                score=0.0,
            )
            for row in rows
        ]

    async def get_observations_after(
        self,
        project: str,
        anchor_time: float,
        limit: int,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
    ) -> List[ObservationIndex]:
        if not self.conn:
            return []
        params: List[Any] = [project, anchor_time]
        conditions = ["project = ?", "created_at > ?"]
        if obs_type:
            conditions.append("type = ?")
            params.append(obs_type)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if date_start is not None:
            conditions.append("created_at >= ?")
            params.append(date_start)
        if date_end is not None:
            conditions.append("created_at <= ?")
            params.append(date_end)
        tag_clause = self._tag_clause(tag_filters, params)
        if tag_clause:
            conditions.append(tag_clause)
        sql = """
            SELECT id, summary, project, type, created_at
            FROM observations
            WHERE {conditions}
            ORDER BY created_at ASC
            LIMIT ?
        """.format(conditions=" AND ".join(conditions))
        params.append(limit)
        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [
            ObservationIndex(
                id=row["id"],
                summary=row["summary"] or "",
                project=row["project"],
                type=row["type"],
                created_at=row["created_at"],
                score=0.0,
            )
            for row in rows
        ]

    def _row_to_observation(
        self, row: sqlite3.Row, assets: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        # NOTE: This method is synchronous but kept mainly as helper when assets are provided.
        # If assets are missing, we return empty list to avoid async complexity in row mapping for now,
        # relying on callers to pre-fetch assets.
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "project": row["project"],
            "type": row["type"],
            "concept": row["concept"] if "concept" in row.keys() else None,
            "title": row["title"],
            "content": row["content"],
            "summary": row["summary"],
            "content_hash": row["content_hash"],
            "created_at": row["created_at"],
            "importance_score": row["importance_score"],
            "tags": json.loads(row["tags"] or "[]"),
            "metadata": json.loads(row["metadata"] or "{}"),
            "diff": row["diff"] if "diff" in row.keys() else None,
            "assets": assets if assets is not None else [],
        }

    def _row_to_session(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "project": row["project"],
            "goal": row["goal"],
            "summary": row["summary"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
        }

    async def increment_access_count(self, obs_id: str) -> None:
        """Increment the access count for an observation.

        Args:
            obs_id: The observation ID to update
        """
        if not self.conn:
            return
        await self.conn.execute(
            """
            UPDATE observations
            SET access_count = COALESCE(access_count, 0) + 1,
                last_accessed_at = ?
            WHERE id = ?
            """,
            (time.time(), obs_id),
        )
        await self.conn.commit()

    async def mark_superseded(
        self,
        obs_id: str,
        superseded_by: str,
    ) -> int:
        """Mark an observation as superseded by another.

        Args:
            obs_id: The observation ID to mark as superseded
            superseded_by: The ID of the observation that supersedes this one

        Returns:
            Number of rows updated
        """
        if not self.conn:
            return 0
        async with self.conn.execute(
            """
            UPDATE observations
            SET superseded_by = ?
            WHERE id = ? AND superseded_by IS NULL
            """,
            (superseded_by, obs_id),
        ) as cursor:
            await self.conn.commit()
            return cursor.rowcount

    async def get_similar_observations(
        self,
        project: str,
        obs_type: Optional[str] = None,
        exclude_superseded: bool = True,
        limit: int = 100,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Get observations for similarity analysis.

        Args:
            project: Project to search in
            obs_type: Optional type filter
            exclude_superseded: Exclude already superseded observations
            limit: Maximum number to return
            date_start: Optional start date filter
            date_end: Optional end date filter

        Returns:
            List of observation dicts with id, content, summary, created_at
        """
        if not self.conn:
            return []
        params: List[Any] = [project]
        conditions = ["project = ?"]
        if obs_type:
            conditions.append("type = ?")
            params.append(obs_type)
        if exclude_superseded:
            conditions.append("superseded_by IS NULL")
        if date_start is not None:
            conditions.append("created_at >= ?")
            params.append(date_start)
        if date_end is not None:
            conditions.append("created_at <= ?")
            params.append(date_end)

        sql = f"""
            SELECT id, content, summary, created_at, type, tags
            FROM observations
            WHERE {" AND ".join(conditions)}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "content": row["content"],
                "summary": row["summary"],
                "created_at": row["created_at"],
                "type": row["type"],
                "tags": json.loads(row["tags"] or "[]"),
            }
            for row in rows
        ]

    async def get_stale_observations(
        self,
        project: str,
        max_age_days: int = 90,
        min_access_count: int = 0,
        exclude_superseded: bool = True,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get observations that are old and rarely accessed (candidates for decay).

        Args:
            project: Project to search in
            max_age_days: Only include observations older than this
            min_access_count: Only include observations with access count <= this
            exclude_superseded: Exclude already superseded observations
            limit: Maximum number to return

        Returns:
            List of observation dicts
        """
        if not self.conn:
            return []
        cutoff_time = time.time() - (max_age_days * 86400)
        params: List[Any] = [project, cutoff_time, min_access_count]
        conditions = [
            "project = ?",
            "created_at < ?",
            "COALESCE(access_count, 0) <= ?",
        ]
        if exclude_superseded:
            conditions.append("superseded_by IS NULL")

        sql = f"""
            SELECT id, content, summary, created_at, type, access_count, importance_score
            FROM observations
            WHERE {" AND ".join(conditions)}
            ORDER BY access_count ASC, created_at ASC
            LIMIT ?
        """
        params.append(limit)

        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "content": row["content"],
                "summary": row["summary"],
                "created_at": row["created_at"],
                "type": row["type"],
                "access_count": row["access_count"] or 0,
                "importance_score": row["importance_score"],
            }
            for row in rows
        ]

    async def get_session_stats(self, project: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.conn:
            return []
        query = """
            SELECT 
                s.id, 
                s.project, 
                s.goal, 
                s.start_time, 
                s.end_time,
                COUNT(o.id) as obs_count,
                MAX(o.created_at) as last_activity
            FROM sessions s
            LEFT JOIN observations o ON s.id = o.session_id
        """
        params: List[Any] = []
        if project:
            query += " WHERE s.project = ?"
            params.append(project)
        
        query += " GROUP BY s.id ORDER BY s.start_time DESC LIMIT ?"
        params.append(limit)
        
        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        
        stats = []
        for row in rows:
            # Calculate duration
            start = row["start_time"]
            end = row["end_time"] or row["last_activity"] or start
            duration = end - start
            
            stats.append({
                "id": row["id"],
                "project": row["project"],
                "goal": row["goal"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "duration": duration,
                "obs_count": row["obs_count"],
            })
        return stats

    # =========================================================================
    # Incremental Indexing Methods
    # =========================================================================

    async def get_unindexed_observations(
        self,
        project: Optional[str] = None,
        limit: int = 100,
        current_version: int = 1,
    ) -> List[Dict[str, Any]]:
        """Get observations that haven't been indexed or need re-indexing.

        Returns observations where:
        - last_indexed_at is NULL (never indexed)
        - OR index_version < current_version (outdated index)

        Args:
            project: Optional project filter
            limit: Maximum number of observations to return
            current_version: Current index version

        Returns:
            List of observation dicts needing indexing
        """
        if not self.conn:
            return []

        params: List[Any] = []
        conditions = [
            "(last_indexed_at IS NULL OR index_version < ?)"
        ]
        params.append(current_version)

        if project:
            conditions.append("project = ?")
            params.append(project)

        where_clause = " WHERE " + " AND ".join(conditions)
        sql = f"""
            SELECT id, session_id, project, type, title, content, summary,
                   content_hash, created_at, importance_score, tags, metadata
            FROM observations
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        async with self.conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    async def mark_observations_indexed(
        self,
        obs_ids: List[str],
        index_version: int = 1,
    ) -> int:
        """Mark observations as indexed.

        Args:
            obs_ids: List of observation IDs to mark
            index_version: Index version number

        Returns:
            Number of observations updated
        """
        if not self.conn or not obs_ids:
            return 0

        now = time.time()
        placeholders = ",".join("?" * len(obs_ids))
        sql = f"""
            UPDATE observations
            SET last_indexed_at = ?, index_version = ?
            WHERE id IN ({placeholders})
        """
        params = [now, index_version] + obs_ids

        async with self.conn.execute(sql, params) as cursor:
            await self.conn.commit()
            return cursor.rowcount

    async def get_indexing_stats(
        self,
        project: Optional[str] = None,
        current_version: int = 1,
    ) -> Dict[str, Any]:
        """Get statistics about indexing status.

        Args:
            project: Optional project filter
            current_version: Current index version

        Returns:
            Dict with indexing statistics
        """
        if not self.conn:
            return {}

        params: List[Any] = []
        where_clause = ""
        if project:
            where_clause = "WHERE project = ?"
            params.append(project)

        # Total observations
        async with self.conn.execute(
            f"SELECT COUNT(*) as count FROM observations {where_clause}",
            params
        ) as cursor:
            row = await cursor.fetchone()
            total = row["count"] if row else 0

        # Never indexed
        params_unindexed = params.copy()
        unindexed_where = "WHERE last_indexed_at IS NULL"
        if project:
            unindexed_where += " AND project = ?"
            params_unindexed.append(project)

        async with self.conn.execute(
            f"SELECT COUNT(*) as count FROM observations {unindexed_where}",
            params_unindexed[len(params):] if project else []
        ) as cursor:
            row = await cursor.fetchone()
            never_indexed = row["count"] if row else 0

        # Outdated index
        params_outdated = [current_version]
        outdated_where = "WHERE last_indexed_at IS NOT NULL AND index_version < ?"
        if project:
            outdated_where += " AND project = ?"
            params_outdated.append(project)

        async with self.conn.execute(
            f"SELECT COUNT(*) as count FROM observations {outdated_where}",
            params_outdated
        ) as cursor:
            row = await cursor.fetchone()
            outdated = row["count"] if row else 0

        # Up to date
        up_to_date = total - never_indexed - outdated

        return {
            "total": total,
            "indexed": up_to_date,
            "never_indexed": never_indexed,
            "outdated": outdated,
            "needs_indexing": never_indexed + outdated,
            "current_version": current_version,
            "index_coverage": (up_to_date / total * 100) if total > 0 else 100.0,
        }

    async def reset_index_status(
        self,
        project: Optional[str] = None,
    ) -> int:
        """Reset indexing status for all observations.

        Useful when rebuilding the entire index.

        Args:
            project: Optional project filter

        Returns:
            Number of observations reset
        """
        if not self.conn:
            return 0

        params: List[Any] = []
        where_clause = ""
        if project:
            where_clause = "WHERE project = ?"
            params.append(project)

        sql = f"""
            UPDATE observations
            SET last_indexed_at = NULL, index_version = 0
            {where_clause}
        """

        async with self.conn.execute(sql, params) as cursor:
            await self.conn.commit()
            return cursor.rowcount

