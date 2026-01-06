import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import ContextConfig
from .memory import MemoryManager

CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def _match_tags(obs_tags: List[str], filters: List[str]) -> bool:
    if not filters:
        return True
    return any(tag in obs_tags for tag in filters)


def _filter_observations(
    observations: List[Dict[str, Any]],
    obs_types: List[str],
    tag_filters: List[str],
) -> List[Dict[str, Any]]:
    filtered = []
    for obs in observations:
        if obs_types and obs.get("type") not in obs_types:
            continue
        if not _match_tags(obs.get("tags") or [], tag_filters):
            continue
        filtered.append(obs)
    return filtered


def _format_index_line(obs: Dict[str, Any], token_count: int, show_tokens: bool) -> str:
    summary = obs.get("summary") or ""
    obs_type = obs.get("type") or "-"
    tags = obs.get("tags") or []
    tag_text = f" | tags: {', '.join(tags)}" if tags else ""
    tokens = f" (~{token_count} tok)" if show_tokens else ""
    return f"- {obs.get('id')} | {obs_type} | {summary}{tag_text}{tokens}"


def _render_header(
    project: str,
    query: Optional[str],
    show_tokens: bool,
    totals: Dict[str, int],
    economics: Optional[Dict[str, float]] = None,
) -> List[str]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# [{project}] context ({timestamp})"]
    if query:
        lines.append(f"Query: {query}")
    if show_tokens:
        lines.append(
            "Token estimate: index ~{index} tok, full ~{full} tok, total ~{total} tok".format(**totals)
        )
        if economics and economics.get("full_total", 0) > 0:
            savings = economics.get("savings", 0.0)
            savings_pct = economics.get("savings_pct", 0.0)
            lines.append(
                "Context economics: full ~{full_total} tok, saved ~{savings} tok ({savings_pct:.1f}%)".format(
                    full_total=int(economics["full_total"]),
                    savings=int(savings),
                    savings_pct=savings_pct,
                )
            )
    lines.append("")
    lines.append(
        "Context Index: summaries for quick recall. Use IDs to pull full details when needed."
    )
    lines.append("")
    return lines


def build_context(
    manager: MemoryManager,
    project: Optional[str] = None,
    query: Optional[str] = None,
    obs_type: Optional[str] = None,
    obs_types: Optional[List[str]] = None,
    tag_filters: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    total_count: Optional[int] = None,
    full_count: Optional[int] = None,
    full_field: Optional[str] = None,
    show_tokens: Optional[bool] = None,
    wrap: Optional[bool] = None,
    config: Optional[ContextConfig] = None,
) -> Tuple[str, Dict[str, Any]]:
    cfg = config or manager.config.context
    total_count = total_count if total_count is not None else cfg.total_observation_count
    full_count = full_count if full_count is not None else cfg.full_observation_count
    if total_count < 0:
        total_count = 0
    if full_count < 0:
        full_count = 0
    obs_type_filters = cfg.observation_types
    if obs_types is not None:
        obs_type_filters = [item for item in obs_types if item]
    if obs_type:
        obs_type_filters = [obs_type]
    tag_filters = tag_filters if tag_filters is not None else cfg.tag_filters
    full_field = (full_field or cfg.full_observation_field).lower()
    if full_field not in {"content", "summary"}:
        full_field = "content"
    show_tokens = cfg.show_token_estimates if show_tokens is None else show_tokens
    wrap = cfg.wrap_context_tag if wrap is None else wrap

    project_name = project or (f"session:{session_id}" if session_id else "all-projects")
    observations: List[Dict[str, Any]] = []

    if query:
        index_items = manager.search(
            query,
            limit=total_count,
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            tag_filters=tag_filters,
        )
        if index_items:
            ids = [item.id for item in index_items]
            details = {item["id"]: item for item in manager.get_observations(ids)}
            for item in index_items:
                detail = details.get(item.id)
                if not detail:
                    continue
                observations.append(detail)
    else:
        observations = manager.db.list_observations(
            project=project if not session_id else None,
            session_id=session_id,
            limit=total_count,
        )

    observations = _filter_observations(observations, obs_type_filters, tag_filters)
    index_observations = observations[:total_count]
    full_observations = observations[:full_count]

    totals = {"index": 0, "full": 0, "total": 0}
    index_lines: List[str] = []
    full_baseline_tokens = 0
    for obs in index_observations:
        summary_text = obs.get("summary") or obs.get("content") or ""
        token_count = estimate_tokens(summary_text)
        totals["index"] += token_count
        full_text = obs.get("content") or summary_text
        full_baseline_tokens += estimate_tokens(full_text)
        index_lines.append(_format_index_line({**obs, "summary": summary_text}, token_count, show_tokens))

    full_lines: List[str] = []
    if full_observations:
        full_lines.append("Full Context:")
        full_lines.append("")

    for obs in full_observations:
        detail_text = obs.get(full_field) or ""
        token_count = estimate_tokens(detail_text)
        totals["full"] += token_count
        full_lines.append(f"## {obs.get('id')} | {obs.get('type') or '-'}")
        full_lines.append(detail_text.strip())
        full_lines.append("")

    totals["total"] = totals["index"] + totals["full"]
    economics = {}
    if show_tokens and full_baseline_tokens > 0:
        savings = max(full_baseline_tokens - totals["index"], 0)
        savings_pct = (savings / full_baseline_tokens) * 100 if full_baseline_tokens else 0.0
        economics = {
            "full_total": full_baseline_tokens,
            "savings": savings,
            "savings_pct": savings_pct,
        }
    lines = _render_header(project_name, query, show_tokens, totals, economics or None)
    lines.extend(index_lines)
    lines.append("")
    lines.extend(full_lines)

    context_text = "\n".join(line for line in lines if line is not None)
    if wrap:
        context_text = f"<ai-mem-context>\n{context_text}\n</ai-mem-context>"

    metadata = {
        "project": project_name,
        "session_id": session_id,
        "query": query,
        "index_count": len(index_observations),
        "full_count": len(full_observations),
        "tokens": totals,
        "economics": economics,
        "full_field": full_field,
        "obs_types": obs_type_filters,
    }
    return context_text, metadata
