import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable
from collections import Counter

from .models import Observation, Session, ObservationIndex


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._configure()
        self.create_tables()

    def _configure(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA journal_mode = WAL")
        self.conn.commit()

    def create_tables(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
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
        cursor.execute(
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
        cursor.execute(
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
        cursor.executescript(
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
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_project ON observations(project)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_type ON observations(type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_created ON observations(created_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_hash ON observations(content_hash)"
        )
        self.conn.commit()
        self._ensure_columns()

    def _tag_clause(
        self,
        tag_filters: Optional[List[str]],
        params: List[Any],
        column: str = "tags",
    ) -> Optional[str]:
        if not tag_filters:
            return None
        clauses = []
        for tag in tag_filters:
            value = str(tag).strip()
            if not value:
                continue
            clauses.append(f"{column} LIKE ?")
            params.append(f'%"{value}"%')
        if not clauses:
            return None
        return "(" + " OR ".join(clauses) + ")"

    def _ensure_columns(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(observations)")
        existing = {row["name"] for row in cursor.fetchall()}
        if "content_hash" not in existing:
            cursor.execute("ALTER TABLE observations ADD COLUMN content_hash TEXT")
            self.conn.commit()

    def add_session(self, session: Session) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
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
        self.conn.commit()

    def add_observation(self, obs: Observation) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
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
                metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        self.conn.commit()

    def find_observation_by_hash(
        self,
        content_hash: str,
        project: str,
    ) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM observations
            WHERE content_hash = ? AND project = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (content_hash, project),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_observation(row)

    def get_observation(self, obs_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM observations WHERE id = ?", (obs_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_observation(row)

    def get_observations(self, obs_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = list(obs_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT * FROM observations WHERE id IN ({placeholders})",
            ids,
        )
        rows = cursor.fetchall()
        return [self._row_to_observation(row) for row in rows]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_session(row)

    def list_sessions(
        self,
        project: Optional[str] = None,
        active_only: bool = False,
        goal_query: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
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
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [self._row_to_session(row) for row in rows]

    def list_observations(
        self,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        params: List[Any] = []
        sql = "SELECT * FROM observations"
        conditions: List[str] = []
        if project:
            conditions.append("project = ?")
            params.append(project)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [self._row_to_observation(row) for row in rows]

    def list_projects(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT project FROM observations ORDER BY project")
        rows = cursor.fetchall()
        return [row["project"] for row in rows]

    def get_stats(
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
        cursor = self.conn.cursor()
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

        cursor.execute(f"SELECT COUNT(*) AS count FROM observations {where_clause}", params)
        total_row = cursor.fetchone()
        total = int(total_row["count"]) if total_row else 0

        end_ts = date_end
        if end_ts is None:
            cursor.execute(
                f"SELECT MAX(created_at) AS max_time FROM observations {where_clause}",
                params,
            )
            max_row = cursor.fetchone()
            if max_row and max_row["max_time"] is not None:
                end_ts = float(max_row["max_time"])

        cursor.execute(
            f"""
            SELECT type, COUNT(*) AS count
            FROM observations
            {where_clause}
            GROUP BY type
            ORDER BY count DESC, type ASC
            """,
            params,
        )
        by_type = [
            {"type": row["type"], "count": int(row["count"])}
            for row in cursor.fetchall()
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
            cursor.execute(
                f"""
                SELECT project, COUNT(*) AS count
                FROM observations
                {project_where}
                GROUP BY project
                ORDER BY count DESC, project ASC
                """,
                project_params,
            )
            by_project = [
                {"project": row["project"], "count": int(row["count"])}
                for row in cursor.fetchall()
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
            cursor.execute(
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
            )
            by_day = [
                {"day": row["day"], "count": int(row["count"])}
                for row in cursor.fetchall()
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
            cursor.execute(
                f"SELECT COUNT(*) AS count FROM observations {prev_where}",
                prev_params,
            )
            prev_row = cursor.fetchone()
            previous_total = int(prev_row["count"]) if prev_row else 0
            trend_delta = recent_total - previous_total
            if previous_total > 0:
                trend_pct = (trend_delta / previous_total) * 100.0

        top_tags: List[Dict[str, Any]] = []
        if tag_limit > 0 and total:
            cursor.execute(
                f"SELECT tags FROM observations {where_clause}",
                params,
            )
            counter: Counter[str] = Counter()
            for row in cursor.fetchall():
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
            cursor.execute(f"SELECT type, tags FROM observations {where_clause}", params)
            per_type: Dict[str, Counter[str]] = {}
            for row in cursor.fetchall():
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

    def delete_observation(self, obs_id: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM observations WHERE id = ?", (obs_id,))
        self.conn.commit()
        return cursor.rowcount

    def update_observation_tags(self, obs_id: str, tags: List[str]) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE observations SET tags = ? WHERE id = ?",
            (json.dumps(tags), obs_id),
        )
        self.conn.commit()
        return cursor.rowcount

    def delete_project(self, project: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM observations WHERE project = ?", (project,))
        self.conn.commit()
        return cursor.rowcount

    def get_tag_counts(
        self,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        obs_type: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
        limit: Optional[int] = 50,
    ) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
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
        cursor.execute(f"SELECT tags FROM observations {where_clause}", params)
        counter = Counter()
        for row in cursor.fetchall():
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

    def replace_tag(
        self,
        old_tag: str,
        new_tag: Optional[str] = None,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        obs_type: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
    ) -> int:
        value = str(old_tag or "").strip()
        if not value:
            return 0
        cursor = self.conn.cursor()
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
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        cursor.execute(f"SELECT id, tags FROM observations {where_clause}", params)
        updated = 0
        replacement = str(new_tag or "").strip() or None
        for row in cursor.fetchall():
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
            cursor.execute(
                "UPDATE observations SET tags = ? WHERE id = ?",
                (json.dumps(deduped), row["id"]),
            )
            updated += 1
        if updated:
            self.conn.commit()
        return updated

    def search_observations_fts(
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
        cursor = self.conn.cursor()
        conditions = []
        params: List[Any] = []

        if query.strip():
            safe_query = query.replace('"', '""')
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

        tag_clause = self._tag_clause(tag_filters, params, column="observations.tags")
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
        cursor.execute(sql, params)
        rows = cursor.fetchall()
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

    def get_recent_observations(
        self,
        project: Optional[str],
        limit: int,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
    ) -> List[ObservationIndex]:
        cursor = self.conn.cursor()
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
        cursor.execute(sql, params)
        rows = cursor.fetchall()
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

    def get_observations_before(
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
        cursor = self.conn.cursor()
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
        cursor.execute(sql, params)
        rows = cursor.fetchall()
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

    def get_observations_after(
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
        cursor = self.conn.cursor()
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
        cursor.execute(sql, params)
        rows = cursor.fetchall()
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

    def _row_to_observation(self, row: sqlite3.Row) -> Dict[str, Any]:
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
