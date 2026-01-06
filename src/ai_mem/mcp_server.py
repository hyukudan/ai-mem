import json
import sys
from typing import Any, Dict, List, Optional

from .context import build_context, estimate_tokens
from .memory import MemoryManager


def _tool_instructions() -> str:
    return (
        "3-LAYER WORKFLOW:\n"
        "1) search(query) -> Get index with IDs\n"
        "2) timeline(anchor=ID) -> Context around results\n"
        "3) get_observations(ids=[...]) -> Full details\n\n"
        "Always filter with search/timeline before fetching full details."
    )


class MCPServer:
    def __init__(self) -> None:
        self.manager = MemoryManager()

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "__IMPORTANT",
                "description": "Workflow documentation for memory search tools.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "search",
                "description": "Search memory index. Params: query, limit, project, session_id, obs_type, date_start, date_end, since, tags, show_tokens.",
                "inputSchema": {"type": "object", "additionalProperties": True},
            },
            {
                "name": "mem-search",
                "description": "Alias for search (natural language memory lookup). Params: query, limit, project, session_id, obs_type, date_start, date_end, since, tags, show_tokens.",
                "inputSchema": {"type": "object", "additionalProperties": True},
            },
            {
                "name": "timeline",
                "description": "Timeline around an observation. Params: anchor or query, depth_before, depth_after, project, session_id, obs_type, date_start, date_end, since, tags, show_tokens.",
                "inputSchema": {"type": "object", "additionalProperties": True},
            },
            {
                "name": "get_observations",
                "description": "Fetch full observation details by IDs. Params: ids (array).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["ids"],
                    "additionalProperties": True,
                },
            },
            {
                "name": "summarize",
                "description": "Summarize recent observations. Params: project, session_id, count, obs_type, store, tags.",
                "inputSchema": {"type": "object", "additionalProperties": True},
            },
            {
                "name": "context",
                "description": (
                    "Build context injection text. Params: project, session_id, query, obs_type, "
                    "obs_types, tags, total, full, full_field, show_tokens, wrap, output."
                ),
                "inputSchema": {"type": "object", "additionalProperties": True},
            },
            {
                "name": "stats",
                "description": "Aggregate stats. Params: project, session_id, obs_type, date_start, date_end, since, tags, tag_limit, day_limit, type_tag_limit.",
                "inputSchema": {"type": "object", "additionalProperties": True},
            },
            {
                "name": "tags",
                "description": "List tags with counts. Params: project, session_id, obs_type, date_start, date_end, tags, limit.",
                "inputSchema": {"type": "object", "additionalProperties": True},
            },
            {
                "name": "tag-add",
                "description": "Add a tag across matching observations. Params: tag, project, session_id, obs_type, date_start, date_end, tags.",
                "inputSchema": {"type": "object", "additionalProperties": True},
            },
            {
                "name": "tag-rename",
                "description": "Rename a tag across matching observations. Params: old_tag, new_tag, project, session_id, obs_type, date_start, date_end, tags.",
                "inputSchema": {"type": "object", "additionalProperties": True},
            },
            {
                "name": "tag-delete",
                "description": "Delete a tag across matching observations. Params: tag, project, session_id, obs_type, date_start, date_end, tags.",
                "inputSchema": {"type": "object", "additionalProperties": True},
            },
        ]

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "__IMPORTANT":
            return self._wrap_text(_tool_instructions())
        if name in {"search", "mem-search"}:
            return self._search(args)
        if name == "timeline":
            return self._timeline(args)
        if name == "get_observations":
            return self._get_observations(args)
        if name == "summarize":
            return self._summarize(args)
        if name == "context":
            return self._context(args)
        if name == "stats":
            return self._stats(args)
        if name == "tags":
            return self._tags(args)
        if name == "tag-add":
            return self._tag_add(args)
        if name == "tag-rename":
            return self._tag_rename(args)
        if name == "tag-delete":
            return self._tag_delete(args)
        return self._wrap_text(f"Unknown tool: {name}", is_error=True)

    @staticmethod
    def _parse_tags(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, list):
            tags = [str(item).strip() for item in value if str(item).strip()]
            return tags or None
        if isinstance(value, str):
            tags = [item.strip() for item in value.split(",") if item.strip()]
            return tags or None
        return None

    @staticmethod
    def _parse_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
        return None

    @staticmethod
    def _parse_list(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            return items or None
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            return items or None
        return None

    def _search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit", 10))
        project = args.get("project")
        session_id = args.get("session_id")
        obs_type = args.get("obs_type") or args.get("type")
        date_start = args.get("date_start")
        date_end = args.get("date_end")
        since = args.get("since")
        tags = self._parse_tags(args.get("tags") or args.get("tag"))
        show_tokens = self._parse_bool(args.get("show_tokens"))
        if session_id:
            project = None
        results = self.manager.search(
            query,
            limit=limit,
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            date_start=date_start,
            date_end=date_end,
            since=since,
            tag_filters=tags,
        )
        payload = [item.model_dump() for item in results]
        if show_tokens:
            for row, item in zip(payload, results):
                row["token_estimate"] = estimate_tokens(item.summary or "")
        return self._wrap_text(json.dumps(payload, indent=2))

    def _timeline(self, args: Dict[str, Any]) -> Dict[str, Any]:
        anchor = args.get("anchor") or args.get("anchor_id")
        query = args.get("query")
        depth_before = int(args.get("depth_before", 3))
        depth_after = int(args.get("depth_after", 3))
        project = args.get("project")
        session_id = args.get("session_id")
        obs_type = args.get("obs_type") or args.get("type")
        date_start = args.get("date_start")
        date_end = args.get("date_end")
        since = args.get("since")
        tags = self._parse_tags(args.get("tags") or args.get("tag"))
        show_tokens = self._parse_bool(args.get("show_tokens"))
        if session_id:
            project = None
        results = self.manager.timeline(
            anchor_id=anchor,
            query=query,
            depth_before=depth_before,
            depth_after=depth_after,
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            date_start=date_start,
            date_end=date_end,
            since=since,
            tag_filters=tags,
        )
        payload = [item.model_dump() for item in results]
        if show_tokens:
            for row, item in zip(payload, results):
                row["token_estimate"] = estimate_tokens(item.summary or "")
        return self._wrap_text(json.dumps(payload, indent=2))

    def _get_observations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ids = args.get("ids") or []
        results = self.manager.get_observations(ids)
        return self._wrap_text(json.dumps(results, indent=2))

    def _summarize(self, args: Dict[str, Any]) -> Dict[str, Any]:
        project = args.get("project")
        session_id = args.get("session_id")
        count = int(args.get("count", 20))
        obs_type = args.get("obs_type") or args.get("type")
        store = args.get("store", True)
        if isinstance(store, str):
            store = store.strip().lower() not in {"false", "0", "no"}
        tags = args.get("tags")
        result = self.manager.summarize_project(
            project=project,
            session_id=session_id,
            limit=count,
            obs_type=obs_type,
            store=bool(store),
            tags=tags if isinstance(tags, list) else None,
        )
        return self._wrap_text(json.dumps(result or {}, indent=2))

    def _context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        project = args.get("project")
        session_id = args.get("session_id")
        query = args.get("query")
        obs_type = args.get("obs_type") or args.get("type")
        obs_types = self._parse_list(args.get("obs_types") or args.get("types"))
        tags = self._parse_tags(args.get("tags") or args.get("tag"))
        total = args.get("total")
        full = args.get("full")
        full_field = args.get("full_field")
        show_tokens = self._parse_bool(args.get("show_tokens"))
        wrap = self._parse_bool(args.get("wrap"))
        output = str(args.get("output") or args.get("format") or "text").strip().lower()
        if session_id:
            project = None
        context_text, meta = build_context(
            self.manager,
            project=project,
            session_id=session_id,
            query=query,
            obs_type=obs_type,
            obs_types=obs_types,
            tag_filters=tags,
            total_count=int(total) if total is not None else None,
            full_count=int(full) if full is not None else None,
            full_field=full_field,
            show_tokens=show_tokens,
            wrap=wrap,
        )
        if output == "json":
            payload = {"context": context_text, "metadata": meta}
            return self._wrap_text(json.dumps(payload, indent=2))
        return self._wrap_text(context_text)

    def _stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        project = args.get("project")
        session_id = args.get("session_id")
        obs_type = args.get("obs_type") or args.get("type")
        date_start = args.get("date_start")
        date_end = args.get("date_end")
        since = args.get("since")
        tags = self._parse_tags(args.get("tags") or args.get("tag"))
        tag_limit = int(args.get("tag_limit", 10))
        day_limit = int(args.get("day_limit", 14))
        type_tag_limit = int(args.get("type_tag_limit", 3))
        if session_id:
            project = None
        results = self.manager.get_stats(
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            date_start=date_start,
            date_end=date_end,
            since=since,
            tag_filters=tags,
            tag_limit=tag_limit,
            day_limit=day_limit,
            type_tag_limit=type_tag_limit,
        )
        return self._wrap_text(json.dumps(results, indent=2))

    def _tags(self, args: Dict[str, Any]) -> Dict[str, Any]:
        project = args.get("project")
        session_id = args.get("session_id")
        obs_type = args.get("obs_type") or args.get("type")
        date_start = args.get("date_start")
        date_end = args.get("date_end")
        tags = self._parse_tags(args.get("tags") or args.get("tag"))
        limit = int(args.get("limit", 50))
        if session_id:
            project = None
        results = self.manager.list_tags(
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tags,
            limit=limit,
        )
        return self._wrap_text(json.dumps(results, indent=2))

    def _tag_add(self, args: Dict[str, Any]) -> Dict[str, Any]:
        value = str(args.get("tag") or args.get("value") or "").strip()
        if not value:
            return self._wrap_text("tag is required", is_error=True)
        project = args.get("project")
        session_id = args.get("session_id")
        obs_type = args.get("obs_type") or args.get("type")
        date_start = args.get("date_start")
        date_end = args.get("date_end")
        tags = self._parse_tags(args.get("tags") or args.get("filter_tags") or args.get("filter_tag"))
        if session_id:
            project = None
        updated = self.manager.add_tag(
            tag=value,
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tags,
        )
        return self._wrap_text(json.dumps({"success": True, "updated": updated}, indent=2))

    def _tag_rename(self, args: Dict[str, Any]) -> Dict[str, Any]:
        old_tag = str(args.get("old_tag") or args.get("from") or "").strip()
        new_tag = str(args.get("new_tag") or args.get("to") or "").strip()
        if not old_tag or not new_tag:
            return self._wrap_text("old_tag and new_tag are required", is_error=True)
        project = args.get("project")
        session_id = args.get("session_id")
        obs_type = args.get("obs_type") or args.get("type")
        date_start = args.get("date_start")
        date_end = args.get("date_end")
        tags = self._parse_tags(args.get("tags") or args.get("filter_tags") or args.get("filter_tag"))
        if session_id:
            project = None
        updated = self.manager.rename_tag(
            old_tag=old_tag,
            new_tag=new_tag,
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tags,
        )
        return self._wrap_text(json.dumps({"success": True, "updated": updated}, indent=2))

    def _tag_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        value = str(args.get("tag") or args.get("value") or "").strip()
        if not value:
            return self._wrap_text("tag is required", is_error=True)
        project = args.get("project")
        session_id = args.get("session_id")
        obs_type = args.get("obs_type") or args.get("type")
        date_start = args.get("date_start")
        date_end = args.get("date_end")
        tags = self._parse_tags(args.get("tags") or args.get("filter_tags") or args.get("filter_tag"))
        if session_id:
            project = None
        updated = self.manager.delete_tag(
            tag=value,
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tags,
        )
        return self._wrap_text(json.dumps({"success": True, "updated": updated}, indent=2))

    @staticmethod
    def _wrap_text(text: str, is_error: bool = False) -> Dict[str, Any]:
        payload = {"content": [{"type": "text", "text": text}]}
        if is_error:
            payload["isError"] = True
        return payload


def run_stdio() -> None:
    server = MCPServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        if method in {"initialize"}:
            result = {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ai-mem-mcp", "version": "0.1.0"},
            }
            _send_response(request_id, result=result)
            continue

        if method in {"tools/list", "list_tools"}:
            _send_response(request_id, result={"tools": server.list_tools()})
            continue

        if method in {"tools/call", "call_tool"}:
            name = params.get("name", "")
            args = params.get("arguments") or {}
            result = server.call_tool(name, args)
            _send_response(request_id, result=result)
            continue

        if method in {"shutdown"}:
            _send_response(request_id, result={})
            break

        _send_response(
            request_id,
            error={"code": -32601, "message": f"Unknown method: {method}"},
        )


def _send_response(
    request_id: Optional[int],
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    if request_id is None:
        return
    payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result or {}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    run_stdio()
