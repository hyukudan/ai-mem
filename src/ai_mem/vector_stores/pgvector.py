import json
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from pgvector import Vector

from .base import VectorStoreProvider

class PGVectorStore(VectorStoreProvider):
    def __init__(
        self,
        dsn: str,
        table: str,
        dimension: int,
        index_type: str = "ivfflat",
        lists: int = 100,
    ):
        self._dsn = dsn
        self._table_name = table
        self._table = sql.Identifier(table)
        self._dimension = max(1, dimension)
        self._index_type = index_type.lower()
        if self._index_type not in {"ivfflat", "hnsw"}:
            raise ValueError("pgvector index type must be 'ivfflat' or 'hnsw'.")
        self._lists = max(1, lists)
        self._conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self._conn)
        self._ensure_table()

    def _ensure_table(self) -> None:
        create_table = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                observation_id TEXT,
                project TEXT,
                session_id TEXT,
                type TEXT,
                created_at DOUBLE PRECISION,
                chunk_index INTEGER,
                metadata JSONB,
                document TEXT,
                embedding vector({dimension})
            )
            """
        ).format(table=self._table, dimension=sql.SQL(str(self._dimension)))
        with self._conn.cursor() as cursor:
            cursor.execute(create_table)
            if self._index_type == "ivfflat":
                idx_sql = sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} ON {table} USING ivfflat (embedding vector_l2_ops) WITH (lists = {lists})"
                ).format(
                index=sql.Identifier(f"{self._table_name}_embedding_idx"),
                    table=self._table,
                    lists=sql.Literal(self._lists),
                )
            else:
                idx_sql = sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} ON {table} USING hnsw (embedding vector_cosine_ops)"
                ).format(
                index=sql.Identifier(f"{self._table_name}_embedding_idx"),
                    table=self._table,
                )
            cursor.execute(idx_sql)

    def _build_where_clause(
        self, where: Optional[Dict[str, Any]]
    ) -> Tuple[sql.SQL, List[Any]]:
        if not where:
            return sql.SQL(""), []
        clauses: List[sql.SQL] = []
        params: List[Any] = []
        for key, value in where.items():
            if value is None:
                continue
            clauses.append(sql.SQL("metadata ->> %s = %s"))
            params.extend([key, str(value)])
        if not clauses:
            return sql.SQL(""), []
        return sql.SQL("WHERE ") + sql.SQL(" AND ").join(clauses), params

    def add(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        if not embeddings:
            return
        rows: List[Tuple[Any, ...]] = []
        for index, embedding in enumerate(embeddings):
            meta = metadatas[index] if index < len(metadatas) else {}
            document = documents[index] if index < len(documents) else ""
            rows.append(
                (
                    ids[index],
                    meta.get("observation_id"),
                    meta.get("project"),
                    meta.get("session_id"),
                    meta.get("type"),
                    float(meta.get("created_at")) if meta.get("created_at") is not None else None,
                    meta.get("chunk_index"),
                    json.dumps(meta),
                    document,
                    Vector(embedding),
                )
            )
        insert_sql = sql.SQL(
            """
            INSERT INTO {table} (
                id,
                observation_id,
                project,
                session_id,
                type,
                created_at,
                chunk_index,
                metadata,
                document,
                embedding
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (id) DO UPDATE SET
                metadata = EXCLUDED.metadata,
                document = EXCLUDED.document,
                embedding = EXCLUDED.embedding
            """
        ).format(table=self._table)
        with self._conn.cursor() as cursor:
            cursor.executemany(insert_sql, rows)

    def query(
        self,
        embedding: List[float],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        where_clause, params = self._build_where_clause(where)
        query_sql = sql.SQL(
            """
            SELECT id, metadata, (embedding <=> %s) AS distance
            FROM {table}
            {where_clause}
            ORDER BY distance ASC
            LIMIT %s
            """
        ).format(table=self._table, where_clause=where_clause or sql.SQL(""))
        args: List[Any] = [Vector(embedding)]
        args.extend(params)
        args.append(n_results)
        with self._conn.cursor() as cursor:
            cursor.execute(query_sql, args)
            rows = cursor.fetchall()
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        distances: List[float] = []
        for row in rows:
            ids.append(row[0])
            metadata_value = row[1] or {}
            if isinstance(metadata_value, str):
                try:
                    metadata_value = json.loads(metadata_value)
                except json.JSONDecodeError:
                    metadata_value = {}
            metadatas.append(metadata_value)
            distances.append(float(row[2]))
        return {
            "ids": [ids],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def delete_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        delete_sql = sql.SQL("DELETE FROM {table} WHERE id = ANY(%s)").format(table=self._table)
        with self._conn.cursor() as cursor:
            cursor.execute(delete_sql, (ids,))

    def delete_where(self, where: Dict[str, Any]) -> None:
        where_clause, params = self._build_where_clause(where)
        delete_sql = sql.SQL("DELETE FROM {table} {where_clause}").format(
            table=self._table, where_clause=where_clause or sql.SQL("")
        )
        with self._conn.cursor() as cursor:
            cursor.execute(delete_sql, params)
