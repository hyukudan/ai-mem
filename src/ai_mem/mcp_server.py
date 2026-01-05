import json
import sys
from typing import Any, Dict, List, Optional

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
                "description": "Search memory index. Params: query, limit, project, obs_type.",
                "inputSchema": {"type": "object", "additionalProperties": True},
            },
            {
                "name": "timeline",
                "description": "Timeline around an observation. Params: anchor or query, depth_before, depth_after, project.",
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
        ]

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "__IMPORTANT":
            return self._wrap_text(_tool_instructions())
        if name == "search":
            return self._search(args)
        if name == "timeline":
            return self._timeline(args)
        if name == "get_observations":
            return self._get_observations(args)
        return self._wrap_text(f"Unknown tool: {name}", is_error=True)

    def _search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit", 10))
        project = args.get("project")
        obs_type = args.get("obs_type") or args.get("type")
        results = self.manager.search(query, limit=limit, project=project, obs_type=obs_type)
        payload = [item.model_dump() for item in results]
        return self._wrap_text(json.dumps(payload, indent=2))

    def _timeline(self, args: Dict[str, Any]) -> Dict[str, Any]:
        anchor = args.get("anchor") or args.get("anchor_id")
        query = args.get("query")
        depth_before = int(args.get("depth_before", 3))
        depth_after = int(args.get("depth_after", 3))
        project = args.get("project")
        results = self.manager.timeline(
            anchor_id=anchor,
            query=query,
            depth_before=depth_before,
            depth_after=depth_after,
            project=project,
        )
        payload = [item.model_dump() for item in results]
        return self._wrap_text(json.dumps(payload, indent=2))

    def _get_observations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ids = args.get("ids") or []
        results = self.manager.get_observations(ids)
        return self._wrap_text(json.dumps(results, indent=2))

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
