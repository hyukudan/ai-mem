import json
import aiosqlite
import sqlite3 # keep for Row type if needed, or row_factory logic
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable

from .models import Observation, Session, ObservationIndex


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self.conn = await aiosqlite.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = aiosqlite.Row
        await self._configure()
        await self.create_tables()

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()

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
        await self.conn.commit()
        await self._ensure_columns()
        await self._ensure_asset_table()

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
            quoted_tags = [f'"{tag.replace("\"", "\"\"")}"' for tag in valid_tags]
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

    async def add_observation(self, obs: Observation) -> None:
        if not self.conn:
            return
        await self.conn.execute(
            """
            INSERT OR REPLACE INTO observations (
                id,
                session_id,
                project,
                type,
                title,
                content,
                summary,
                content_hash,
                created_at,
                importance_score,
                tags,
                metadata,
                diff
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                obs.id,
                obs.session_id,
                obs.project,
                obs.type,
                obs.title,
                obs.content,
                obs.summary,
                obs.content_hash,
                obs.created_at,
                obs.importance_score,
                json.dumps(obs.tags),
                json.dumps(obs.metadata),
                obs.diff,
            ),
        )
        await self.conn.commit()

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
    ) -> List[ObservationIndex]:
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
    ) -> List[ObservationIndex]:
        if not self.conn:
            return []
        conditions = []
        params: List[Any] = []

        if query.strip():
            # Escape double quotes and wrap in phrase markers to avoid FTS operator injection
            safe_query = '"' + query.replace('"', '""') + '"'
            conditions.append("observations_fts MATCH ?")
            params.append(safe_query)

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
    ) -> List[ObservationIndex]:
        if not self.conn:
            return []
        params: List[Any] = []
        sql = """
            SELECT id, summary, project, type, created_at
            FROM observations
        """
        conditions = []
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

