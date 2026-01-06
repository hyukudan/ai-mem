import asyncio
import csv
import io
import json
import os
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

import uvicorn

from .context import build_context, estimate_tokens
from . import __version__
from .memory import MemoryManager

app = FastAPI(title="ai-mem Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_manager: Optional[MemoryManager] = None
_api_token = os.environ.get("AI_MEM_API_TOKEN")


def get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


def _check_token(request: Request, query_token: Optional[str] = None) -> None:
    if not _api_token:
        return
    auth = request.headers.get("authorization") or ""
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    if not token:
        token = request.headers.get("x-ai-mem-token", "")
    if not token and query_token:
        token = query_token
    if token != _api_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _parse_list_param(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


class MemoryInput(BaseModel):
    content: str
    obs_type: str = "note"
    project: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    title: Optional[str] = None
    summarize: bool = True


class ObservationIds(BaseModel):
    ids: List[str]


class ObservationUpdate(BaseModel):
    tags: Optional[List[str]] = None


class ProjectDelete(BaseModel):
    project: str


class ImportPayload(BaseModel):
    items: List[Dict[str, Any]]
    project: Optional[str] = None


class SummarizeRequest(BaseModel):
    project: Optional[str] = None
    session_id: Optional[str] = None
    count: int = 20
    obs_type: Optional[str] = None
    store: bool = True
    tags: List[str] = Field(default_factory=list)


class SessionStartRequest(BaseModel):
    project: Optional[str] = None
    goal: Optional[str] = None
    session_id: Optional[str] = None


class SessionEndRequest(BaseModel):
    session_id: Optional[str] = None
    project: Optional[str] = None


class TagRenameRequest(BaseModel):
    old_tag: str
    new_tag: str
    project: Optional[str] = None
    session_id: Optional[str] = None
    obs_type: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    tags: Optional[str] = None


class TagDeleteRequest(BaseModel):
    tag: str
    project: Optional[str] = None
    session_id: Optional[str] = None
    obs_type: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    tags: Optional[str] = None


class TagAddRequest(BaseModel):
    tag: str
    project: Optional[str] = None
    session_id: Optional[str] = None
    obs_type: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    tags: Optional[str] = None


class ContextRequest(BaseModel):
    project: Optional[str] = None
    session_id: Optional[str] = None
    query: Optional[str] = None
    obs_type: Optional[str] = None
    obs_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    total: Optional[int] = None
    full: Optional[int] = None
    full_field: Optional[str] = None
    show_tokens: Optional[bool] = None
    wrap: Optional[bool] = None


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ai-mem Viewer</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
            :root {
                color-scheme: light;
                --bg: #f4f1ea;
                --panel: #ffffff;
                --ink: #1d1d1b;
                --muted: #6f6a61;
                --accent: #1f6f59;
                --accent-2: #e36b2c;
                --shadow: 0 18px 50px rgba(18, 18, 18, 0.12);
            }
            body {
                font-family: "Space Grotesk", sans-serif;
                margin: 0;
                min-height: 100vh;
                background:
                    radial-gradient(circle at 15% 20%, #fbe7d8, transparent 45%),
                    radial-gradient(circle at 85% 15%, #dde9ff, transparent 40%),
                    linear-gradient(180deg, var(--bg) 0%, #fbf8f3 100%);
                color: var(--ink);
            }
            .page {
                max-width: 1200px;
                margin: 0 auto;
                padding: 36px 24px 64px;
                display: grid;
                grid-template-columns: 280px 1fr;
                gap: 24px;
            }
            header {
                grid-column: 1 / -1;
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: var(--panel);
                padding: 18px 24px;
                border-radius: 16px;
                box-shadow: var(--shadow);
            }
            h1 { font-size: 28px; margin: 0; font-weight: 700; }
            .subtitle { color: var(--muted); font-size: 14px; }
            .panel {
                background: var(--panel);
                border-radius: 18px;
                padding: 18px;
                box-shadow: var(--shadow);
            }
            .controls { display: grid; gap: 12px; }
            label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
            select, input {
                width: 100%;
                padding: 10px 12px;
                border-radius: 10px;
                border: 1px solid #e1dbd2;
                font-size: 14px;
                background: #fffdf8;
            }
            .input-with-action {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .input-with-action input {
                flex: 1;
            }
            .input-with-action .query-clear {
                padding: 8px 12px;
                border-radius: 10px;
                border: 1px solid #e1dbd2;
                background: #f1ede4;
                color: #3a3731;
                font-size: 12px;
                font-weight: 600;
                min-width: 64px;
            }
            .input-with-action .query-clear:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .query-options {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 11px;
                color: var(--muted);
            }
            .query-options label {
                font-size: 11px;
                text-transform: none;
                letter-spacing: 0;
                color: var(--muted);
            }
            button {
                padding: 12px 16px;
                border-radius: 12px;
                border: none;
                font-weight: 600;
                cursor: pointer;
                background: var(--accent);
                color: #fff;
            }
            .secondary {
                background: #f1ede4;
                color: #3a3731;
            }
            .actions { display: grid; gap: 8px; }
            .actions .row { display: grid; gap: 8px; }
            .actions .row.inline { grid-template-columns: 1fr 1fr; }
            .row.inline {
                display: inline-flex;
                gap: 8px;
                align-items: center;
            }
            .quick-row {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
            }
            .auto-row {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 10px;
                font-size: 12px;
                color: var(--muted);
            }
            .auto-row .mode {
                display: inline-flex;
                align-items: center;
                gap: 6px;
            }
            .auto-row .live-indicator {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-weight: 600;
                color: var(--muted);
            }
            .auto-row .live-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #b8b2a8;
                box-shadow: 0 0 0 0 rgba(31, 111, 89, 0.0);
                transition: background 0.2s ease;
            }
            .auto-row .live-indicator.active {
                color: var(--accent);
            }
            .auto-row .live-indicator.active .live-dot {
                background: var(--accent);
                box-shadow: 0 0 0 6px rgba(31, 111, 89, 0.15);
            }
            .auto-row .auto-mode {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                font-size: 11px;
                font-weight: 600;
                color: var(--muted);
            }
            .auto-row .auto-mode.global {
                color: var(--accent);
            }
            .auto-row .auto-mode.global::before {
                content: "●";
                font-size: 8px;
                margin-right: 2px;
            }
            .auto-row .auto-global {
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .pulse-toggle {
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .auto-row input[type="number"] {
                width: 70px;
                padding: 6px 8px;
                border-radius: 8px;
                border: 1px solid #e1dbd2;
                background: #fffdf8;
                font-size: 12px;
            }
            .chip {
                padding: 6px 10px;
                border-radius: 999px;
                border: 1px solid #e1dbd2;
                background: #fff7ec;
                color: #3a3731;
                font-weight: 600;
                font-size: 12px;
            }
            .chip.warn {
                border-color: #e1b7a6;
                color: #b23b21;
                background: #fbe9e4;
            }
            .limit-warning {
                font-size: 11px;
                color: #b23b21;
                display: none;
            }
            .results {
                display: grid;
                gap: 12px;
            }
            .results-header {
                display: none;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                padding: 10px 12px;
                margin-bottom: 12px;
                border-radius: 12px;
                background: #f7f1e6;
                color: var(--muted);
                font-size: 12px;
                position: sticky;
                top: 0;
                z-index: 2;
                box-shadow: 0 10px 24px rgba(30, 30, 30, 0.08);
            }
            .results-header .filter-icons {
                display: inline-flex;
                gap: 4px;
                margin-left: 6px;
                vertical-align: middle;
            }
            .results-header .filter-icon {
                width: 10px;
                height: 10px;
                font-size: 7px;
            }
            .results-header .pulse-status {
                margin-left: 6px;
                font-size: 11px;
                color: #b23b21;
                font-weight: 600;
            }
            .pulse-status {
                margin-left: 6px;
                font-size: 10px;
                color: #b23b21;
                font-weight: 600;
            }
            .results-header .live-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                display: inline-block;
                background: #b8b2a8;
                margin-left: 6px;
            }
            .results-header.live .live-dot {
                background: var(--accent);
                box-shadow: 0 0 0 4px rgba(31, 111, 89, 0.2);
                animation: pulse 1.6s ease-in-out infinite;
            }
            body.pulse-off .results-header.live .live-dot {
                animation: none;
            }
            .filters-pill {
                display: none;
                align-items: center;
                gap: 6px;
                padding: 4px 10px;
                border-radius: 999px;
                background: #efe5d8;
                color: #6b5f50;
                font-size: 11px;
                font-weight: 600;
            }
            .filters-pill button {
                padding: 2px 6px;
                border-radius: 999px;
                font-size: 10px;
            }
            .filters-pill .filter-icon {
                width: 14px;
                height: 14px;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 8px;
                font-weight: 700;
                color: #fff;
            }
            .filters-pill .filter-icon.project { background: #1f6f59; }
            .filters-pill .filter-icon.type { background: #e36b2c; }
            .filters-pill .filter-icon.date { background: #2f6fb0; }
            .filters-pill .filter-icon.query { background: #9a6ea6; }
            .filters-pill .filter-icon.tag { background: #2a8a6a; }
            .filters-pill .filter-icon.session { background: #5c7c3a; }
            .filters-pill .live-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                display: inline-block;
                background: #b8b2a8;
                margin-left: 4px;
            }
            .filters-pill.live .live-dot {
                background: var(--accent);
                box-shadow: 0 0 0 4px rgba(31, 111, 89, 0.2);
                animation: pulse 1.6s ease-in-out infinite;
            }
            body.pulse-off .filters-pill.live .live-dot {
                animation: none;
            }
            .timeline-status {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 6px 10px;
                border-radius: 999px;
                background: #efe5d8;
                color: #6b5f50;
                font-size: 11px;
                font-weight: 600;
            }
            .timeline-status .timeline-value {
                max-width: 160px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .timeline-status strong {
                color: var(--ink);
                font-weight: 600;
            }
            .timeline-status button {
                padding: 2px 6px;
                border-radius: 999px;
                font-size: 10px;
            }
            .timeline-badge {
                margin-left: 6px;
                padding: 2px 6px;
                border-radius: 999px;
                background: #efe5d8;
                color: #6b5f50;
                font-size: 10px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }
            .auto-mode-badge {
                margin-left: 6px;
                padding: 2px 6px;
                border-radius: 999px;
                background: #e6f3ee;
                color: #1f6f59;
                font-size: 10px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .anchor-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 8px;
                border-radius: 999px;
                border: 1px solid #e1dbd2;
                background: #fff7ec;
                color: var(--muted);
                font-size: 11px;
            }
            .anchor-badge strong {
                color: var(--ink);
                font-weight: 600;
            }
            .anchor-badge .anchor-value {
                max-width: 140px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .anchor-badge button {
                border: none;
                background: transparent;
                color: var(--muted);
                font-size: 12px;
                cursor: pointer;
                padding: 0;
            }
            .anchor-badge button:hover {
                color: var(--ink);
            }
            .card {
                background: var(--panel);
                padding: 16px;
                border-radius: 16px;
                box-shadow: 0 12px 30px rgba(30, 30, 30, 0.08);
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .card:hover {
                transform: translateY(-2px);
                box-shadow: 0 16px 40px rgba(30, 30, 30, 0.12);
            }
            .meta {
                color: var(--muted);
                font-size: 0.8em;
                margin-top: 8px;
            }
            .detail {
                background: var(--panel);
                border-radius: 18px;
                padding: 18px;
                box-shadow: var(--shadow);
                display: grid;
                gap: 8px;
            }
            .detail pre {
                white-space: pre-wrap;
                background: #f5f2ea;
                padding: 12px;
                border-radius: 12px;
            }
            .detail .button-row {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            .tag-editor {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                align-items: center;
            }
            .tag-editor input {
                flex: 1;
                min-width: 180px;
            }
            .pill {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: #f5e8da;
                color: #7a4a2f;
                font-size: 12px;
                margin-right: 6px;
            }
            .pill.clickable {
                cursor: pointer;
            }
            .pill.clickable:hover {
                background: #f1dcc5;
            }
            .accent { color: var(--accent-2); }
            .token-row {
                display: grid;
                gap: 8px;
                padding: 12px;
                border-radius: 12px;
                background: #f7f1e6;
            }
            .stats-card {
                padding: 12px;
                border-radius: 12px;
                background: #f3efe8;
                display: grid;
                gap: 8px;
            }
            .context-card {
                padding: 12px;
                border-radius: 12px;
                background: #efe6d8;
                display: grid;
                gap: 10px;
            }
            .saved-card {
                padding: 12px;
                border-radius: 12px;
                background: #f3efe8;
                display: grid;
                gap: 10px;
            }
            .saved-title {
                font-weight: 600;
                color: var(--muted);
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 0.08em;
            }
            .saved-row {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                align-items: center;
            }
            .saved-row input {
                flex: 1;
                min-width: 120px;
            }
            .tag-card {
                padding: 12px;
                border-radius: 12px;
                background: #f7f1e6;
                display: grid;
                gap: 10px;
            }
            .tag-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
            }
            .tag-title {
                font-weight: 600;
                color: var(--muted);
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 0.08em;
            }
            .tag-meta {
                font-size: 11px;
                color: var(--muted);
            }
            .tag-list {
                display: grid;
                gap: 6px;
                max-height: 140px;
                overflow: auto;
            }
            .tag-item {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
            }
            .tag-count {
                font-size: 11px;
                color: var(--muted);
            }
            .tag-editor {
                display: grid;
                gap: 8px;
            }
            .tag-row {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            .session-card {
                padding: 12px;
                border-radius: 12px;
                background: #f1ede4;
                display: grid;
                gap: 8px;
            }
            .session-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
            }
            .session-title {
                font-weight: 600;
                color: var(--muted);
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 0.08em;
            }
            .session-status {
                font-size: 12px;
                color: var(--muted);
            }
            .session-list {
                display: grid;
                gap: 6px;
                max-height: 160px;
                overflow: auto;
            }
            .session-item {
                padding: 8px;
                border-radius: 10px;
                background: #f8f5ef;
                display: grid;
                gap: 4px;
            }
            .session-item .meta {
                margin: 0;
            }
            .session-goal {
                font-size: 12px;
                color: var(--muted);
            }
            .session-summary {
                font-size: 12px;
                color: var(--muted);
            }
            .session-actions {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            .session-stats {
                display: grid;
                gap: 6px;
                font-size: 12px;
                color: var(--muted);
            }
            .session-stats .stat-row {
                display: flex;
                justify-content: space-between;
                gap: 10px;
            }
            .session-row {
                display: grid;
                gap: 6px;
            }
            .stream-card {
                padding: 12px;
                border-radius: 12px;
                background: #eef0e6;
                display: grid;
                gap: 8px;
            }
            .stream-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
            }
            .stream-title {
                font-weight: 600;
                color: var(--muted);
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 0.08em;
            }
            .stream-status {
                font-size: 12px;
                color: var(--muted);
            }
            .stream-status.connected {
                color: var(--accent);
            }
            .stream-list {
                display: grid;
                gap: 6px;
                max-height: 220px;
                overflow: auto;
            }
            .stream-item {
                padding: 8px;
                border-radius: 10px;
                background: #f7f4ec;
                cursor: pointer;
                transition: background 0.15s ease;
            }
            .stream-item:hover {
                background: #efe8dc;
            }
            .stream-meta {
                font-size: 11px;
                color: var(--muted);
                margin-top: 4px;
                display: flex;
                align-items: center;
                gap: 8px;
                flex-wrap: wrap;
            }
            .stream-actions {
                display: flex;
                justify-content: flex-end;
            }
            .stream-row {
                display: grid;
                gap: 6px;
            }
            .context-title {
                font-weight: 600;
                color: var(--muted);
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 0.08em;
            }
            .context-grid {
                display: grid;
                gap: 8px;
            }
            .context-row {
                display: grid;
                gap: 6px;
            }
            .context-inline {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 13px;
                color: var(--muted);
            }
            .context-actions {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            .stats-body {
                display: grid;
                gap: 8px;
                max-height: 1000px;
                transition: max-height 0.3s ease, opacity 0.3s ease;
                overflow: hidden;
            }
            .stats-card.collapsed .stats-body {
                max-height: 0;
                opacity: 0;
            }
            @media (prefers-reduced-motion: reduce) {
                .stats-body {
                    transition: none;
                }
                .stats-title.live .live-dot {
                    animation: none;
                }
                .results-header.live .live-dot {
                    animation: none;
                }
                .filters-pill.live .live-dot {
                    animation: none;
                }
            }
            .stats-card.collapsed #stats,
            .stats-card.collapsed #updateHistory,
            .stats-card.collapsed .anchor-pill,
            .stats-card.collapsed .timeline-status {
                display: none;
            }
            .stats-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
            }
            .stats-meta {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                margin-left: auto;
                flex-wrap: wrap;
                justify-content: flex-end;
            }
            .live-badge {
                display: none;
                align-items: center;
                gap: 6px;
                font-size: 10px;
                font-weight: 700;
                background: #e6f3ee;
                color: #1f6f59;
                border-radius: 999px;
                padding: 2px 8px;
            }
            .live-badge.off {
                background: #f6e7e3;
                color: #b23b21;
            }
            @media (max-width: 960px) {
                .stats-header {
                    position: sticky;
                    top: 0;
                    background: #f3efe8;
                    padding: 8px 0;
                    z-index: 1;
                }
            }
            .stats-toggle {
                border: none;
                background: transparent;
                color: var(--accent);
                font-weight: 600;
                cursor: pointer;
            }
            .stats-title {
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                font-size: 11px;
                color: var(--muted);
            }
            .stats-title .live-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                display: inline-block;
                background: #b8b2a8;
                margin-left: 6px;
            }
            .stats-title.live .live-dot {
                background: var(--accent);
                box-shadow: 0 0 0 4px rgba(31, 111, 89, 0.2);
                animation: pulse 1.6s ease-in-out infinite;
            }
            body.pulse-off .stats-title.live .live-dot {
                animation: none;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(31, 111, 89, 0.35); }
                70% { box-shadow: 0 0 0 6px rgba(31, 111, 89, 0.0); }
                100% { box-shadow: 0 0 0 0 rgba(31, 111, 89, 0.0); }
            }
            .last-update {
                font-size: 11px;
                color: var(--muted);
            }
            .anchor-pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-size: 11px;
                color: var(--muted);
                background: #f6efe3;
                border-radius: 999px;
                padding: 4px 8px;
            }
            .anchor-pill button {
                border: none;
                background: transparent;
                color: var(--accent);
                cursor: pointer;
                padding: 0;
                font-weight: 600;
            }
            .update-history {
                font-size: 11px;
                color: var(--muted);
                display: grid;
                gap: 2px;
            }
            .stat-row {
                display: flex;
                justify-content: space-between;
                gap: 12px;
                font-size: 13px;
            }
            .stat-row strong {
                color: var(--accent);
            }
            .trend {
                font-weight: 600;
            }
            .trend.up {
                color: var(--accent);
            }
            .trend.down {
                color: #b23b21;
            }
            .trend.flat {
                color: var(--muted);
            }
            .stat-section {
                margin-top: 6px;
                font-size: 12px;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .project-bars {
                display: grid;
                gap: 6px;
            }
            .project-bar {
                display: grid;
                grid-template-columns: 90px 1fr 32px;
                align-items: center;
                gap: 8px;
                font-size: 12px;
                color: var(--ink);
            }
            .project-bar span {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .project-bar .bar-track {
                height: 10px;
                background: #efe7db;
                border-radius: 999px;
                position: relative;
            }
            .project-bar .bar-fill {
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, #1f6f59, #4aa37f);
            }
            .mini-chart {
                display: grid;
                grid-auto-flow: column;
                align-items: end;
                gap: 6px;
                height: 80px;
                padding: 8px 6px;
                border-radius: 10px;
                border: 1px solid #f0e5d2;
                background: #fff7ec;
            }
            .mini-bar {
                width: 10px;
                border-radius: 6px 6px 2px 2px;
                background: linear-gradient(180deg, var(--accent-2), #f7b37b);
                box-shadow: 0 6px 12px rgba(227, 107, 44, 0.2);
            }
            @media (max-width: 960px) {
                .page { grid-template-columns: 1fr; }
                header {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 8px;
                }
                .results-header {
                    flex-direction: column;
                    align-items: flex-start;
                    padding: 8px 10px;
                    gap: 8px;
                }
                .pulse-help {
                    display: none;
                }
                .results-header.live .live-dot,
                .stats-title.live .live-dot,
                .filters-pill.live .live-dot {
                    box-shadow: 0 0 0 2px rgba(31, 111, 89, 0.2);
                    animation-duration: 2.2s;
                }
                .pulse-status {
                    display: none;
                }
                .filters-pill button {
                    display: none;
                }
                .timeline-status {
                    font-size: 10px;
                    padding: 4px 8px;
                }
                .timeline-status .timeline-value {
                    display: none;
                }
                .timeline-badge {
                    font-size: 9px;
                    padding: 2px 5px;
                }
                .auto-mode-badge {
                    font-size: 9px;
                    padding: 2px 5px;
                }
                .actions .row.inline {
                    grid-template-columns: 1fr;
                }
                .card {
                    padding: 12px;
                    border-radius: 14px;
                }
                .meta {
                    font-size: 0.75em;
                }
            }
        </style>
    </head>
    <body>
        <div class="page">
            <header>
                <div>
                    <h1>ai-mem Viewer</h1>
                    <div class="subtitle">Search and review memory across projects.</div>
                </div>
                <div class="subtitle">API: /api/search, /api/timeline, /api/context, /api/observations</div>
            </header>
            <aside class="panel">
                <div class="controls">
                    <div>
                        <label for="query">Query</label>
                        <div class="input-with-action">
                            <input type="text" id="query" placeholder="Search memories...">
                            <button type="button" class="query-clear" id="queryClear" onclick="clearQuery()">Clear</button>
                        </div>
                        <div class="query-options">
                            <label class="pulse-toggle">
                                <input type="checkbox" id="queryGlobal">
                                Use global query
                            </label>
                        </div>
                    </div>
                    <div>
                        <label for="project">Project</label>
                        <select id="project"></select>
                    </div>
                    <div>
                        <label for="sessionId">Session ID (overrides project)</label>
                        <input type="text" id="sessionId" placeholder="optional">
                    </div>
                    <div>
                        <label for="type">Type</label>
                        <select id="type">
                            <option value="">Any</option>
                            <option value="decision">decision</option>
                            <option value="bugfix">bugfix</option>
                            <option value="feature">feature</option>
                            <option value="refactor">refactor</option>
                            <option value="discovery">discovery</option>
                            <option value="change">change</option>
                            <option value="note">note</option>
                            <option value="interaction">interaction</option>
                            <option value="tool_output">tool_output</option>
                            <option value="file_content">file_content</option>
                            <option value="summary">summary</option>
                        </select>
                    </div>
                    <div>
                        <label for="tagsFilter">Tags</label>
                        <input type="text" id="tagsFilter" placeholder="comma,separated">
                    </div>
                    <div class="saved-card" id="savedFiltersCard">
                        <div class="saved-title">Saved filters</div>
                        <select id="savedFilters">
                            <option value="">Select saved filter</option>
                        </select>
                        <div class="saved-row">
                            <input type="text" id="savedFilterName" placeholder="name">
                            <button class="secondary" onclick="saveCurrentFilter()">Save</button>
                        </div>
                        <div class="saved-row">
                            <button class="secondary" onclick="applySavedFilter()">Apply</button>
                            <button class="secondary" onclick="deleteSavedFilter()">Delete</button>
                        </div>
                    </div>
                    <div class="tag-card" id="tagCard">
                        <div class="tag-header">
                            <div class="tag-title">Tags</div>
                            <button class="chip" onclick="loadTagManager()">Refresh</button>
                        </div>
                        <div class="tag-meta" id="tagMeta">Scope: current filters</div>
                        <div class="tag-list" id="tagList"></div>
                        <div class="tag-editor">
                            <input type="text" id="tagAdd" placeholder="add tag">
                            <input type="text" id="tagOld" placeholder="tag">
                            <input type="text" id="tagNew" placeholder="rename to">
                        </div>
                        <div class="tag-row">
                            <button class="secondary" onclick="addTag()">Add</button>
                            <button class="secondary" onclick="renameTag()">Rename</button>
                            <button class="secondary" onclick="deleteTag()">Delete</button>
                        </div>
                    </div>
                    <div class="context-card" id="contextCard">
                        <div class="context-title">Context</div>
                        <div class="context-grid">
                            <div class="context-row">
                                <label for="contextTotal">Index count</label>
                                <input type="number" id="contextTotal" min="1" max="100" value="12">
                            </div>
                            <div class="context-row">
                                <label for="contextFull">Full count</label>
                                <input type="number" id="contextFull" min="0" max="50" value="4">
                            </div>
                            <div class="context-row">
                                <label for="contextField">Full field</label>
                                <select id="contextField">
                                    <option value="content">content</option>
                                    <option value="summary">summary</option>
                                </select>
                            </div>
                            <div class="context-row">
                                <label for="contextTags">Tags</label>
                                <input type="text" id="contextTags" placeholder="comma,separated">
                            </div>
                            <div class="context-row">
                                <label for="contextTypes">Types</label>
                                <input type="text" id="contextTypes" placeholder="note,bugfix,summary">
                            </div>
                            <label class="context-inline">
                                <input type="checkbox" id="contextTokens" checked>
                                Show token estimates
                            </label>
                            <label class="context-inline">
                                <input type="checkbox" id="contextWrap" checked>
                                Wrap &lt;ai-mem-context&gt;
                            </label>
                            <label class="context-inline">
                                <input type="checkbox" id="contextGlobal">
                                Use global context settings
                            </label>
                            <div class="context-actions">
                                <button class="chip" onclick="openContextPreview()">Preview</button>
                                <button class="chip" onclick="copyContext()">Copy</button>
                                <button class="chip" onclick="summarizeContext()">Summarize</button>
                            </div>
                        </div>
                    </div>
                    <div class="session-card" id="sessionCard">
                        <div class="session-header">
                            <div class="session-title">Sessions</div>
                            <label class="pulse-toggle">
                                <input type="checkbox" id="sessionShowAll">
                                Show all
                            </label>
                        </div>
                        <div class="session-status" id="sessionStatus">No active session</div>
                        <div class="session-row">
                            <label for="sessionGoal">Goal (optional)</label>
                            <input type="text" id="sessionGoal" placeholder="e.g. fix OAuth flow">
                        </div>
                        <div class="session-row">
                            <label for="sessionGoalFilter">Goal filter</label>
                            <input type="text" id="sessionGoalFilter" placeholder="contains...">
                        </div>
                        <div class="session-row">
                            <label for="sessionDateStart">Start after</label>
                            <input type="date" id="sessionDateStart">
                        </div>
                        <div class="session-row">
                            <label for="sessionDateEnd">Start before</label>
                            <input type="date" id="sessionDateEnd">
                        </div>
                        <div class="session-actions">
                            <button class="chip" onclick="startSession()">Start</button>
                            <button class="secondary" onclick="endLatestSession()">End latest</button>
                            <button class="secondary" onclick="refreshSessions()">Refresh</button>
                        </div>
                        <div class="session-list" id="sessionList">
                            <div class="subtitle">No sessions loaded.</div>
                        </div>
                    </div>
                    <div class="stats-card" id="statsCard">
                        <div class="stats-header">
                            <div class="stats-title" id="statsTitle">Stats<span class="live-dot" id="statsLiveDot"></span></div>
                            <div class="stats-meta">
                                <div class="anchor-badge" id="anchorBadge" style="display:none;"></div>
                                <div class="filters-pill" id="filtersPillStats" style="display:none;">Filters</div>
                                <span class="live-badge" id="sidebarLiveBadge">Live</span>
                                <button class="stats-toggle" id="statsToggle" onclick="toggleStats()">▾ Collapse</button>
                            </div>
                        </div>
                        <div class="stats-body" id="statsBody">
                            <div class="last-update" id="lastUpdate">Last update: --</div>
                            <div class="update-history" id="updateHistory"></div>
                            <div class="anchor-pill" id="anchorPill" style="display:none;">
                                <span id="anchorLabel">Timeline anchor set</span>
                                <button onclick="copyTimelineAnchor()">Copy</button>
                                <button onclick="clearTimelineAnchor()">Clear</button>
                            </div>
                            <div id="stats"></div>
                        </div>
                    </div>
                    <div class="stream-card" id="streamCard">
                        <div class="stream-header">
                            <div class="stream-title">Live stream</div>
                            <label class="pulse-toggle">
                                <input type="checkbox" id="streamToggle">
                                Stream
                            </label>
                        </div>
                        <div class="stream-status" id="streamStatus">Stream off</div>
                        <div class="stream-row">
                            <label for="streamQuery">Query filter</label>
                            <input type="text" id="streamQuery" placeholder="optional">
                        </div>
                        <div class="stream-list" id="streamList">
                            <div class="subtitle">No events yet.</div>
                        </div>
                        <div class="stream-actions">
                            <button class="secondary" onclick="exportStreamJson()">Export JSON</button>
                            <button class="secondary" onclick="exportStreamCsv()">Export CSV</button>
                            <button class="secondary" onclick="clearStream()">Clear</button>
                        </div>
                    </div>
                    <div class="filters-pill" id="filtersPillSidebar" style="display:none;">Filters active</div>
                    <div class="timeline-status" id="timelineStatus" style="display:none;"></div>
                    <div class="token-row">
                        <label for="token">API token (optional)</label>
                        <input type="password" id="token" placeholder="Set token for API requests">
                        <button class="secondary" onclick="saveToken()">Save token</button>
                    </div>
                    <div>
                        <label for="limit">Limit</label>
                        <input type="number" id="limit" min="1" max="100" value="10">
                        <div class="limit-warning" id="limitWarning">Limit above 100 may be slow.</div>
                    </div>
                    <div class="quick-row">
                        <button class="chip" onclick="setListLimit(10)">10</button>
                        <button class="chip" onclick="setListLimit(25)">25</button>
                        <button class="chip" onclick="setListLimit(50)">50</button>
                        <button class="chip warn" onclick="setListLimit(100)">100</button>
                    </div>
                    <div class="quick-row">
                        <button class="chip" onclick="setTimelineDepth(1, 1)">±1</button>
                        <button class="chip" onclick="setTimelineDepth(3, 3)">±3</button>
                        <button class="chip" onclick="setTimelineDepth(5, 5)">±5</button>
                    </div>
                    <div>
                        <label for="depthBefore">Timeline depth before</label>
                        <input type="number" id="depthBefore" min="0" max="20" value="3">
                    </div>
                    <div>
                        <label for="depthAfter">Timeline depth after</label>
                        <input type="number" id="depthAfter" min="0" max="20" value="3">
                    </div>
                    <div>
                        <label for="dateStart">Date start</label>
                        <input type="date" id="dateStart">
                    </div>
                    <div>
                        <label for="dateEnd">Date end</label>
                        <input type="date" id="dateEnd">
                    </div>
                    <div class="quick-row">
                        <button class="chip" onclick="setLastDays(7)">Last 7</button>
                        <button class="chip" onclick="setLastDays(30)">Last 30</button>
                        <button class="chip" onclick="clearDates()">All time</button>
                        <button class="chip" onclick="clearAllFilters()">Clear filters</button>
                    </div>
                    <div class="auto-row">
                        <label>
                            <input type="checkbox" id="autoRefresh">
                            Auto-refresh
                        </label>
                        <label>
                            Interval (s)
                            <input type="number" id="refreshInterval" min="5" value="30">
                        </label>
                        <label class="mode">
                            <input type="radio" name="refreshMode" value="all" checked>
                            Stats + results
                        </label>
                        <label class="mode">
                            <input type="radio" name="refreshMode" value="stats">
                            Stats only
                        </label>
                        <label class="auto-global">
                            <input type="checkbox" id="autoGlobal">
                            Use global auto-refresh
                        </label>
                        <label class="pulse-toggle">
                            <input type="checkbox" id="pulseToggle" checked>
                            Pulse
                        </label>
                        <label class="pulse-toggle">
                            <input type="checkbox" id="pulseGlobal">
                            Use global
                        </label>
                        <span class="subtitle pulse-help">Pulse animates live indicators.</span>
                        <span class="live-indicator" id="liveIndicator">
                            <span class="live-dot"></span>
                            Live
                        </span>
                        <span class="auto-mode" id="autoModeLabel"></span>
                    </div>
                    <div class="actions">
                        <div class="row">
                            <button onclick="search()">Search</button>
                            <button class="secondary" onclick="timeline()">Timeline</button>
                        </div>
                        <div class="row inline">
                            <button class="secondary" onclick="exportData()">Export</button>
                            <button class="secondary" onclick="triggerImport()">Import</button>
                        </div>
                        <div class="row inline">
                            <button class="secondary" onclick="exportStats()">Stats CSV</button>
                            <button class="secondary" onclick="exportStatsJson()">Stats JSON</button>
                        </div>
                        <div class="row">
                            <button class="secondary" onclick="deleteProject()">Delete project</button>
                        </div>
                    </div>
                </div>
            </aside>
            <main class="panel">
                <div class="results-header" id="resultsHeader"></div>
                <div class="filters-pill" id="filtersPill">Filters active</div>
                <div class="results" id="results"></div>
            </main>
            <section class="detail" id="detail">
                <div class="subtitle">Select a result to view details.</div>
            </section>
        </div>
        <input type="file" id="importFile" accept="application/json" style="display:none" onchange="importFile(event)">

        <script>
            let currentObservationId = null;
            let autoRefreshTimer = null;
            let lastMode = 'search';
            let updateHistory = [];
            let typingPauseUntil = 0;
            let timelineAnchorId = '';
            let timelineQuery = '';
            let timelineDepthBefore = '3';
            let timelineDepthAfter = '3';
            let listLimit = '10';
            let selectedProject = '';
            let selectedSessionId = '';
            let selectedType = '';
            let selectedTags = '';
            let selectedDateStart = '';
            let selectedDateEnd = '';
            let streamSource = null;
            let streamItems = [];
            const streamMaxItems = 20;
            let savedFilters = [];

            async function loadProjects() {
                const response = await fetch('/api/projects', { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                const projects = await response.json();
                const select = document.getElementById('project');
                const current = selectedProject || select.value;
                select.innerHTML = '<option value="">All projects</option>';
                projects.forEach(project => {
                    const option = document.createElement('option');
                    option.value = project;
                    option.textContent = project;
                    select.appendChild(option);
                });
                if (current && projects.includes(current)) {
                    select.value = current;
                }
                await loadStats();
                await loadTagManager();
                loadAutoRefresh();
                loadPulseToggle();
                loadLastMode();
                loadTimelineAnchor();
                loadQuery();
                await loadContextConfig();
                refreshSessions();
                restartStreamIfActive();
            }

            function buildQueryParams() {
                const query = document.getElementById('query').value || " ";
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const type = document.getElementById('type').value;
                const tags = (document.getElementById('tagsFilter').value || '').trim();
                const limit = document.getElementById('limit').value;
                listLimit = limit || listLimit;
                localStorage.setItem('ai-mem-list-limit', listLimit);
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                const params = new URLSearchParams({ query });
                if (sessionId) {
                    params.append('session_id', sessionId);
                } else if (project) {
                    params.append('project', project);
                }
                if (type) params.append('obs_type', type);
                if (tags) params.append('tags', tags);
                if (limit) params.append('limit', limit);
                if (dateStart) params.append('date_start', dateStart);
                if (dateEnd) params.append('date_end', dateEnd);
                return params.toString();
            }

            function buildContextParams() {
                const params = new URLSearchParams();
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const query = document.getElementById('query').value;
                const type = document.getElementById('type').value;
                if (sessionId) {
                    params.append('session_id', sessionId);
                } else if (project) {
                    params.append('project', project);
                }
                if (query) params.append('query', query);
                if (type) params.append('obs_type', type);

                const config = readContextForm();
                if (config.total) params.append('total', String(config.total));
                if (config.full !== null && config.full !== undefined) params.append('full', String(config.full));
                if (config.fullField) params.append('full_field', config.fullField);
                if (config.tags) params.append('tags', config.tags);
                if (config.types) params.append('obs_types', config.types);
                params.append('show_tokens', config.showTokens ? 'true' : 'false');
                params.append('wrap', config.wrap ? 'true' : 'false');
                return params;
            }

            function openContextPreview() {
                const params = buildContextParams();
                const url = `/api/context/preview?${params.toString()}`;
                fetch(url, { headers: getAuthHeaders() })
                    .then(async response => {
                        if (await handleAuthError(response)) return;
                        if (!response.ok) {
                            alert('Failed to fetch context preview');
                            return;
                        }
                        const text = await response.text();
                        const blob = new Blob([text], { type: 'text/plain' });
                        const blobUrl = URL.createObjectURL(blob);
                        window.open(blobUrl, '_blank', 'noopener');
                        setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
                    })
                    .catch(err => {
                        console.error(err);
                        alert('Failed to fetch context preview');
                    });
            }

            async function copyContext() {
                const params = buildContextParams();
                const response = await fetch(`/api/context/inject?${params.toString()}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    alert('Failed to fetch context');
                    return;
                }
                const text = await response.text();
                try {
                    await navigator.clipboard.writeText(text);
                    alert('Context copied');
                } catch (err) {
                    console.error(err);
                    alert('Copy failed');
                }
            }

            async function summarizeContext() {
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                if (!project && !sessionId) {
                    alert('Select a project or session to summarize.');
                    return;
                }
                const type = document.getElementById('type').value;
                const config = readContextForm();
                const payload = {
                    project: sessionId ? null : (project || null),
                    session_id: sessionId || null,
                    count: config.total || 20,
                    obs_type: type || null,
                    store: true,
                };
                const response = await fetch('/api/summarize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        ...getAuthHeaders(),
                    },
                    body: JSON.stringify(payload),
                });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    alert('Failed to summarize');
                    return;
                }
                const data = await response.json();
                if (data.status !== 'ok') {
                    alert('No observations found to summarize.');
                    return;
                }
                const detail = document.getElementById('detail');
                const obs = data.observation || {};
                const scopeLabel = sessionId ? `Session ${sessionId}` : (project || 'Global');
                detail.innerHTML = `
                    <h2>Summary</h2>
                    <div class="meta">${escapeHtml(scopeLabel)} • ${escapeHtml(obs.id || '')}</div>
                    <pre>${escapeHtml(data.summary || '')}</pre>
                `;
            }

            function buildStatsParams(options = {}) {
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const type = document.getElementById('type').value;
                const tags = (document.getElementById('tagsFilter').value || '').trim();
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                const params = new URLSearchParams();
                if (sessionId) {
                    params.append('session_id', sessionId);
                } else if (project) {
                    params.append('project', project);
                }
                if (type) params.append('obs_type', type);
                if (tags) params.append('tags', tags);
                if (dateStart) params.append('date_start', dateStart);
                if (dateEnd) params.append('date_end', dateEnd);
                params.append('day_limit', String(options.day_limit ?? 7));
                params.append('tag_limit', String(options.tag_limit ?? 7));
                if (options.type_tag_limit !== undefined) {
                    params.append('type_tag_limit', String(options.type_tag_limit));
                }
                return params;
            }

            function safeSlug(value, fallback) {
                const text = (value || '').toString().trim();
                if (!text) return fallback;
                const slug = text.toLowerCase().replace(/[^a-z0-9]+/gi, '_').replace(/^_+|_+$/g, '');
                return slug || fallback;
            }

            function csvEscape(value) {
                const text = value === null || value === undefined ? '' : String(value);
                if (/[",\n]/.test(text)) {
                    return `"${text.replace(/"/g, '""')}"`;
                }
                return text;
            }

            function escapeHtml(value) {
                const text = value === null || value === undefined ? '' : String(value);
                return text
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');
            }

            function truncateText(value, limit) {
                const text = value === null || value === undefined ? '' : String(value);
                if (text.length <= limit) return text;
                const trimmed = text.slice(0, Math.max(0, limit - 3)).trim();
                return trimmed ? `${trimmed}...` : text.slice(0, limit);
            }

            function estimateTokens(text) {
                if (!text) return 0;
                return Math.max(1, Math.ceil(text.length / 4));
            }

            function formatLocalDate(date) {
                const year = date.getFullYear();
                const month = String(date.getMonth() + 1).padStart(2, '0');
                const day = String(date.getDate()).padStart(2, '0');
                return `${year}-${month}-${day}`;
            }

            function formatTime(date) {
                const hours = String(date.getHours()).padStart(2, '0');
                const minutes = String(date.getMinutes()).padStart(2, '0');
                const seconds = String(date.getSeconds()).padStart(2, '0');
                return `${hours}:${minutes}:${seconds}`;
            }

            function formatStreamTime(value) {
                if (!value && value !== 0) return '';
                const date = new Date(Number(value) * 1000);
                if (Number.isNaN(date.getTime())) return '';
                return formatTime(date);
            }

            function buildStreamParams() {
                const params = new URLSearchParams();
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const type = document.getElementById('type').value;
                const tags = (document.getElementById('tagsFilter').value || '').trim();
                const streamQuery = (document.getElementById('streamQuery')?.value || '').trim();
                const token = localStorage.getItem('ai-mem-token') || '';
                if (sessionId) {
                    params.append('session_id', sessionId);
                } else if (project) {
                    params.append('project', project);
                }
                if (type) params.append('obs_type', type);
                if (tags) params.append('tags', tags);
                if (streamQuery) params.append('query', streamQuery);
                if (token) params.append('token', token);
                return params;
            }

            function updateStreamStatus(text, connected) {
                const status = document.getElementById('streamStatus');
                if (!status) return;
                status.textContent = text;
                status.classList.toggle('connected', !!connected);
            }

            function renderStreamItems() {
                const list = document.getElementById('streamList');
                if (!list) return;
                list.innerHTML = '';
                if (!streamItems.length) {
                    list.innerHTML = '<div class="subtitle">No events yet.</div>';
                    return;
                }
                streamItems.forEach(item => {
                    const card = document.createElement('div');
                    card.className = 'stream-item';
                    card.addEventListener('click', () => {
                        if (item.id) {
                            loadDetail(item.id);
                        }
                    });
                    const summary = document.createElement('div');
                    summary.textContent = item.summary || '(no summary)';
                    const meta = document.createElement('div');
                    meta.className = 'stream-meta';
                    const timestamp = formatStreamTime(item.created_at);
                    const parts = [
                        item.project || 'Global',
                        item.type || 'note',
                        timestamp || '',
                        item.token_estimate ? `~${item.token_estimate} tok` : '',
                    ].filter(Boolean);
                    meta.textContent = parts.join(' • ');
                    if (item.session_id) {
                        const sessionBtn = document.createElement('button');
                        sessionBtn.className = 'secondary';
                        sessionBtn.textContent = 'Session';
                        sessionBtn.addEventListener('click', event => {
                            event.stopPropagation();
                            loadSessionDetail(item.session_id);
                        });
                        meta.appendChild(sessionBtn);
                    }
                    card.appendChild(summary);
                    card.appendChild(meta);
                    list.appendChild(card);
                });
            }

            function addStreamItem(item) {
                streamItems = [item, ...streamItems].slice(0, streamMaxItems);
                renderStreamItems();
            }

            function exportStreamJson() {
                if (!streamItems.length) {
                    alert('No stream events to export.');
                    return;
                }
                const blob = new Blob([JSON.stringify(streamItems, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `ai-mem-stream-${Date.now()}.json`;
                document.body.appendChild(link);
                link.click();
                link.remove();
                URL.revokeObjectURL(url);
            }

            function exportStreamCsv() {
                if (!streamItems.length) {
                    alert('No stream events to export.');
                    return;
                }
                const rows = [['id', 'summary', 'project', 'session_id', 'type', 'created_at', 'tags', 'token_estimate']];
                streamItems.forEach(item => {
                    rows.push([
                        item.id || '',
                        item.summary || '',
                        item.project || '',
                        item.session_id || '',
                        item.type || '',
                        item.created_at || '',
                        (item.tags || []).join(' '),
                        item.token_estimate || '',
                    ]);
                });
                const csv = rows.map(row => row.map(csvEscape).join(',')).join('\n');
                const blob = new Blob([csv], { type: 'text/csv' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `ai-mem-stream-${Date.now()}.csv`;
                document.body.appendChild(link);
                link.click();
                link.remove();
                URL.revokeObjectURL(url);
            }

            function clearStream() {
                streamItems = [];
                renderStreamItems();
            }

            function startStream() {
                stopStream();
                const params = buildStreamParams();
                const url = `/api/stream?${params.toString()}`;
                streamSource = new EventSource(url);
                updateStreamStatus('Connecting...', false);
                streamSource.onopen = () => updateStreamStatus('Connected', true);
                streamSource.onmessage = event => {
                    if (!event.data) return;
                    try {
                        const payload = JSON.parse(event.data);
                        addStreamItem(payload);
                    } catch (err) {
                        console.warn('Stream parse failed', err);
                    }
                };
                streamSource.onerror = () => {
                    updateStreamStatus('Reconnecting...', false);
                };
            }

            function stopStream() {
                if (streamSource) {
                    streamSource.close();
                    streamSource = null;
                }
                updateStreamStatus('Stream off', false);
            }

            function toggleStream() {
                const toggle = document.getElementById('streamToggle');
                if (!toggle) return;
                localStorage.setItem('ai-mem-stream-enabled', String(toggle.checked));
                if (toggle.checked) {
                    startStream();
                } else {
                    stopStream();
                }
            }

            function restartStreamIfActive() {
                const toggle = document.getElementById('streamToggle');
                if (toggle && toggle.checked) {
                    startStream();
                }
            }

            function loadStreamToggle() {
                const toggle = document.getElementById('streamToggle');
                if (!toggle) return;
                const stored = localStorage.getItem('ai-mem-stream-enabled');
                toggle.checked = stored === 'true';
                if (toggle.checked) {
                    startStream();
                } else {
                    updateStreamStatus('Stream off', false);
                }
            }

            function loadStreamQuery() {
                const input = document.getElementById('streamQuery');
                if (!input) return;
                const stored = localStorage.getItem('ai-mem-stream-query') || '';
                input.value = stored;
            }

            function loadSessionFilters() {
                const toggle = document.getElementById('sessionShowAll');
                if (!toggle) return;
                const stored = localStorage.getItem('ai-mem-sessions-show-all');
                toggle.checked = stored === 'true';
                const goal = document.getElementById('sessionGoalFilter');
                if (goal) {
                    goal.value = localStorage.getItem('ai-mem-sessions-goal') || '';
                }
                const dateStart = document.getElementById('sessionDateStart');
                if (dateStart) {
                    dateStart.value = localStorage.getItem('ai-mem-sessions-date-start') || '';
                }
                const dateEnd = document.getElementById('sessionDateEnd');
                if (dateEnd) {
                    dateEnd.value = localStorage.getItem('ai-mem-sessions-date-end') || '';
                }
            }

            function updateLastUpdate() {
                const target = document.getElementById('lastUpdate');
                if (!target) return;
                const timeLabel = formatTime(new Date());
                target.textContent = `Last update: ${timeLabel}`;
                updateHistory = [timeLabel, ...updateHistory.filter(item => item !== timeLabel)].slice(0, 3);
                const list = document.getElementById('updateHistory');
                if (list) {
                    list.innerHTML = updateHistory.map(item => `<span>${item}</span>`).join('');
                }
            }

            function toggleStats() {
                const card = document.getElementById('statsCard');
                const toggle = document.getElementById('statsToggle');
                if (!card || !toggle) return;
                const collapsed = card.classList.toggle('collapsed');
                toggle.textContent = collapsed ? '▸ Expand' : '▾ Collapse';
                localStorage.setItem('ai-mem-stats-collapsed', String(collapsed));
            }

            function loadStatsCollapse() {
                const card = document.getElementById('statsCard');
                const toggle = document.getElementById('statsToggle');
                if (!card || !toggle) return;
                const stored = localStorage.getItem('ai-mem-stats-collapsed');
                const collapsed = stored === 'true' || (stored === null && window.innerWidth <= 960);
                card.classList.toggle('collapsed', collapsed);
                toggle.textContent = collapsed ? '▸ Expand' : '▾ Collapse';
            }

            async function refreshSessions() {
                const project = getCurrentProjectValue();
                const params = new URLSearchParams();
                const showAllToggle = document.getElementById('sessionShowAll');
                const showAll = showAllToggle ? showAllToggle.checked : false;
                const goalFilter = (document.getElementById('sessionGoalFilter')?.value || '').trim();
                const dateStart = document.getElementById('sessionDateStart')?.value || '';
                const dateEnd = document.getElementById('sessionDateEnd')?.value || '';
                if (project) params.append('project', project);
                params.append('active_only', showAll ? 'false' : 'true');
                params.append('limit', showAll ? '20' : '10');
                if (goalFilter) params.append('goal', goalFilter);
                if (dateStart) params.append('date_start', dateStart);
                if (dateEnd) params.append('date_end', dateEnd);
                const response = await fetch(`/api/sessions?${params.toString()}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    updateSessionStatus('Failed to load sessions.');
                    return;
                }
                const data = await response.json();
                renderSessions(Array.isArray(data) ? data : [], showAll);
            }

            function updateSessionStatus(text) {
                const status = document.getElementById('sessionStatus');
                if (status) status.textContent = text;
            }

            function renderSessions(items, showAll) {
                const list = document.getElementById('sessionList');
                if (!list) return;
                list.innerHTML = '';
                if (!items.length) {
                    updateSessionStatus(showAll ? 'No sessions' : 'No active session');
                    list.innerHTML = '<div class="subtitle">No active sessions.</div>';
                    return;
                }
                if (showAll) {
                    const activeCount = items.filter(item => !item.end_time).length;
                    updateSessionStatus(`${items.length} session${items.length === 1 ? '' : 's'} (${activeCount} active)`);
                } else {
                    updateSessionStatus(`${items.length} active session${items.length === 1 ? '' : 's'}`);
                }
                items.forEach(item => {
                    const id = item.id || '';
                    const shortId = id ? `${id.slice(0, 8)}...` : '(unknown)';
                    const ended = !!item.end_time;
                    const card = document.createElement('div');
                    card.className = 'session-item';
                    const title = document.createElement('div');
                    title.textContent = `Session ${shortId}${ended ? ' (ended)' : ''}`;
                    if (id) title.title = id;
                    const meta = document.createElement('div');
                    meta.className = 'meta';
                    const start = item.start_time ? formatStreamTime(item.start_time) : '';
                    const end = item.end_time ? formatStreamTime(item.end_time) : '';
                    const project = item.project || 'Global';
                    const timeBits = [start ? `Start ${start}` : '', end ? `End ${end}` : ''].filter(Boolean).join(' • ');
                    meta.textContent = [project, timeBits].filter(Boolean).join(' • ');
                    if (item.goal) {
                        const goal = document.createElement('div');
                        goal.className = 'session-goal';
                        goal.textContent = item.goal;
                        card.appendChild(goal);
                    }
                    if (item.summary) {
                        const summary = document.createElement('div');
                        summary.className = 'session-summary';
                        summary.textContent = truncateText(item.summary, 140);
                        card.appendChild(summary);
                    }
                    const actions = document.createElement('div');
                    actions.className = 'session-actions';
                    const viewBtn = document.createElement('button');
                    viewBtn.className = 'secondary';
                    viewBtn.textContent = 'View';
                    viewBtn.addEventListener('click', () => loadSessionDetail(id));
                    const endBtn = document.createElement('button');
                    endBtn.className = 'secondary';
                    endBtn.textContent = 'End';
                    endBtn.disabled = ended;
                    if (!ended) {
                        endBtn.addEventListener('click', () => endSessionById(id));
                    }
                    actions.appendChild(viewBtn);
                    actions.appendChild(endBtn);
                    card.appendChild(title);
                    card.appendChild(meta);
                    card.appendChild(actions);
                    list.appendChild(card);
                });
            }

            async function startSession() {
                const project = getCurrentProjectValue() || '';
                const goalInput = document.getElementById('sessionGoal');
                const payload = {
                    project: project || null,
                    goal: goalInput ? goalInput.value.trim() : '',
                };
                const response = await fetch('/api/sessions/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                    body: JSON.stringify(payload),
                });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    alert('Failed to start session.');
                    return;
                }
                if (goalInput) goalInput.value = '';
                refreshSessions();
            }

            async function endLatestSession() {
                const project = getCurrentProjectValue() || '';
                const response = await fetch('/api/sessions/end', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                    body: JSON.stringify({ project: project || null }),
                });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    alert('No active session to end.');
                    return;
                }
                refreshSessions();
            }

            async function endSessionById(id) {
                if (!id) return;
                const response = await fetch('/api/sessions/end', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                    body: JSON.stringify({ session_id: id }),
                });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    alert('Failed to end session.');
                    return;
                }
                refreshSessions();
            }

            async function loadSessionDetail(id) {
                if (!id) return;
                const sessionResponse = await fetch(`/api/sessions/${id}`, { headers: getAuthHeaders() });
                if (await handleAuthError(sessionResponse)) return;
                if (!sessionResponse.ok) {
                    alert('Failed to load session.');
                    return;
                }
                const session = await sessionResponse.json();
                const obsResponse = await fetch(`/api/sessions/${id}/observations?limit=50`, { headers: getAuthHeaders() });
                if (await handleAuthError(obsResponse)) return;
                const observations = obsResponse.ok ? await obsResponse.json() : [];
                const statsResponse = await fetch(`/api/stats?session_id=${id}&tag_limit=5&day_limit=14&type_tag_limit=3`, { headers: getAuthHeaders() });
                if (await handleAuthError(statsResponse)) return;
                const stats = statsResponse.ok ? await statsResponse.json() : null;
                const detail = document.getElementById('detail');
                const start = session.start_time ? formatStreamTime(session.start_time) : '';
                const end = session.end_time ? formatStreamTime(session.end_time) : '';
                const project = session.project || 'Global';
                const goal = session.goal ? `<div class="meta">Goal: ${escapeHtml(session.goal)}</div>` : '';
                const sessionSummary = session.summary
                    ? `<div class="accent">Summary</div><pre>${escapeHtml(session.summary)}</pre>`
                    : '';
                let statsHtml = '';
                if (stats) {
                    const total = stats.total || 0;
                    const byType = (stats.by_type || [])
                        .map(item => `<div class="stat-row"><span>${escapeHtml(item.type || '-')}</span><strong>${item.count || 0}</strong></div>`)
                        .join('');
                    const topTags = (stats.top_tags || [])
                        .map(item => `<span>${escapeHtml(item.tag || '-')} (${item.count || 0})</span>`)
                        .join(' • ');
                    statsHtml = `
                        <div class="accent">Session stats</div>
                        <div class="session-stats">
                            <div class="stat-row"><span>Total observations</span><strong>${total}</strong></div>
                            ${byType ? `<div>${byType}</div>` : ''}
                            ${topTags ? `<div class="meta">Top tags: ${topTags}</div>` : ''}
                        </div>
                    `;
                }
                const rows = (observations || []).map(obs => {
                    const summary = escapeHtml(obs.summary || obs.content || '(no summary)');
                    const meta = `${escapeHtml(obs.type || 'note')} • ${escapeHtml(obs.id || '')}`;
                    return `<div class="card" onclick="loadDetail('${obs.id}')"><div>${summary}</div><div class="meta">${meta}</div></div>`;
                }).join('');
                const listHtml = rows || '<div class="subtitle">No observations in this session.</div>';
                detail.innerHTML = `
                    <div><strong>Session ${escapeHtml(id)}</strong></div>
                    <div class="meta">${escapeHtml(project)} • ${start ? `Start ${start}` : ''}${end ? ` • End ${end}` : ''}</div>
                    ${goal}
                    ${sessionSummary}
                    ${statsHtml}
                    <div class="accent">Observations</div>
                    <div class="button-row">
                        <label class="pulse-toggle">
                            <input type="checkbox" id="sessionSummaryStore" checked>
                            Store summary
                        </label>
                        <button class="secondary" onclick="summarizeSession('${id}')">Summarize session</button>
                        <button class="secondary" onclick="exportSession('${id}')">Export session</button>
                    </div>
                    <div class="results">${listHtml}</div>
                `;
            }

            async function exportSession(id) {
                if (!id) return;
                const params = new URLSearchParams({ session_id: id });
                const response = await fetch(`/api/export?${params.toString()}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    alert('Failed to export session.');
                    return;
                }
                const data = await response.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `ai-mem-session-${id}.json`;
                document.body.appendChild(link);
                link.click();
                link.remove();
                URL.revokeObjectURL(url);
            }

            async function summarizeSession(id) {
                if (!id) return;
                const storeToggle = document.getElementById('sessionSummaryStore');
                const store = storeToggle ? storeToggle.checked : false;
                const response = await fetch('/api/summarize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                    body: JSON.stringify({ session_id: id, count: 50, store }),
                });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    alert('Failed to summarize session.');
                    return;
                }
                const data = await response.json();
                if (data.status === 'empty') {
                    alert('No observations to summarize.');
                    return;
                }
                const detail = document.getElementById('detail');
                detail.innerHTML = `
                    <div><strong>Session summary</strong></div>
                    <pre>${escapeHtml(data.summary || '')}</pre>
                    <div class="button-row">
                        <button class="secondary" onclick="loadSessionDetail('${id}')">Back to session</button>
                    </div>
                `;
            }

            function getCurrentProjectValue() {
                const select = document.getElementById('project');
                if (select) {
                    const value = select.value;
                    if (value || select.options.length > 0) return value;
                }
                return selectedProject || localStorage.getItem('ai-mem-selected-project') || '';
            }

            function getTimelineKeys(projectValue) {
                const key = encodeURIComponent(projectValue || 'all');
                return {
                    id: `ai-mem-timeline-anchor-id-${key}`,
                    query: `ai-mem-timeline-anchor-query-${key}`,
                };
            }

            function getQueryKey(projectValue) {
                const key = encodeURIComponent(projectValue || 'all');
                return `ai-mem-query-${key}`;
            }

            function getQueryUseGlobalKey(projectValue) {
                const key = encodeURIComponent(projectValue || 'all');
                return `ai-mem-query-use-global-${key}`;
            }

            function getQueryConfig(projectValue) {
                const key = getQueryKey(projectValue);
                const useGlobalKey = getQueryUseGlobalKey(projectValue);
                const projectStored = localStorage.getItem(key);
                const globalStored = localStorage.getItem('ai-mem-query') || '';
                const useGlobalStored = localStorage.getItem(useGlobalKey);
                const useGlobal = useGlobalStored === null
                    ? projectStored === null
                    : useGlobalStored === 'true';
                const value = useGlobal
                    ? globalStored
                    : (projectStored === null ? globalStored : projectStored);
                return { value, useGlobal };
            }

            function getModeKey(projectValue) {
                const key = encodeURIComponent(projectValue || 'all');
                return `ai-mem-last-mode-${key}`;
            }

            function getAutoRefreshKeys(projectValue) {
                const key = encodeURIComponent(projectValue || 'all');
                return {
                    enabled: `ai-mem-auto-refresh-${key}`,
                    interval: `ai-mem-auto-refresh-interval-${key}`,
                    mode: `ai-mem-auto-refresh-mode-${key}`,
                    useGlobal: `ai-mem-auto-refresh-use-global-${key}`,
                };
            }

            const DEFAULT_CONTEXT_CONFIG = {
                total: 12,
                full: 4,
                fullField: 'content',
                showTokens: true,
                wrap: true,
                tags: '',
                types: '',
            };
            let contextDefaults = { ...DEFAULT_CONTEXT_CONFIG };

            function getContextKeys(projectValue) {
                const key = encodeURIComponent(projectValue || 'all');
                return {
                    config: `ai-mem-context-${key}`,
                    useGlobal: `ai-mem-context-use-global-${key}`,
                };
            }

            function parseContextConfig(value) {
                if (!value) return {};
                try {
                    const parsed = JSON.parse(value);
                    if (parsed && typeof parsed === 'object') return parsed;
                } catch (err) {
                    console.warn('Failed to parse context config', err);
                }
                return {};
            }

            async function loadContextDefaults() {
                try {
                    const response = await fetch('/api/context/config', { headers: getAuthHeaders() });
                    if (await handleAuthError(response)) return;
                    if (!response.ok) return;
                    const data = await response.json();
                    contextDefaults = {
                        total: data.total_observation_count ?? contextDefaults.total,
                        full: data.full_observation_count ?? contextDefaults.full,
                        fullField: data.full_observation_field ?? contextDefaults.fullField,
                        showTokens: data.show_token_estimates ?? contextDefaults.showTokens,
                        wrap: data.wrap_context_tag ?? contextDefaults.wrap,
                        tags: (data.tag_filters || []).join(','),
                        types: (data.observation_types || []).join(','),
                    };
                } catch (err) {
                    console.warn('Failed to load context defaults', err);
                }
            }

            function getContextConfig(projectValue) {
                const keys = getContextKeys(projectValue);
                const projectStored = localStorage.getItem(keys.config);
                const globalStored = localStorage.getItem('ai-mem-context') || '';
                const useGlobalStored = localStorage.getItem(keys.useGlobal);
                const useGlobal = useGlobalStored === null
                    ? projectStored === null
                    : useGlobalStored === 'true';
                const storedText = useGlobal ? globalStored : (projectStored === null ? globalStored : projectStored);
                const stored = parseContextConfig(storedText);
                return { ...contextDefaults, ...stored, useGlobal };
            }

            function readContextForm() {
                const totalInput = document.getElementById('contextTotal');
                const fullInput = document.getElementById('contextFull');
                const fieldInput = document.getElementById('contextField');
                const tagsInput = document.getElementById('contextTags');
                const typesInput = document.getElementById('contextTypes');
                const tokensToggle = document.getElementById('contextTokens');
                const wrapToggle = document.getElementById('contextWrap');
                const globalToggle = document.getElementById('contextGlobal');

                const total = parseInt(totalInput?.value || '', 10);
                const full = parseInt(fullInput?.value || '', 10);
                const fullField = fieldInput?.value || contextDefaults.fullField;
                const tags = (tagsInput?.value || '').trim();
                const types = (typesInput?.value || '').trim();
                const showTokens = tokensToggle ? tokensToggle.checked : contextDefaults.showTokens;
                const wrap = wrapToggle ? wrapToggle.checked : contextDefaults.wrap;
                const useGlobal = globalToggle ? globalToggle.checked : false;

                return {
                    total: Number.isFinite(total) ? total : contextDefaults.total,
                    full: Number.isFinite(full) ? full : contextDefaults.full,
                    fullField,
                    tags,
                    types,
                    showTokens,
                    wrap,
                    useGlobal,
                };
            }

            function applyContextConfig(config) {
                const totalInput = document.getElementById('contextTotal');
                const fullInput = document.getElementById('contextFull');
                const fieldInput = document.getElementById('contextField');
                const tagsInput = document.getElementById('contextTags');
                const typesInput = document.getElementById('contextTypes');
                const tokensToggle = document.getElementById('contextTokens');
                const wrapToggle = document.getElementById('contextWrap');
                const globalToggle = document.getElementById('contextGlobal');
                if (totalInput) totalInput.value = String(config.total ?? contextDefaults.total);
                if (fullInput) fullInput.value = String(config.full ?? contextDefaults.full);
                if (fieldInput) fieldInput.value = config.fullField || contextDefaults.fullField;
                if (tagsInput) tagsInput.value = config.tags || '';
                if (typesInput) typesInput.value = config.types || '';
                if (tokensToggle) tokensToggle.checked = !!config.showTokens;
                if (wrapToggle) wrapToggle.checked = !!config.wrap;
                if (globalToggle) globalToggle.checked = !!config.useGlobal;
            }

            function persistContextConfig(config, projectValue) {
                const keys = getContextKeys(projectValue);
                const payload = {
                    total: config.total,
                    full: config.full,
                    fullField: config.fullField,
                    showTokens: config.showTokens,
                    wrap: config.wrap,
                    tags: config.tags,
                    types: config.types,
                };
                if (config.useGlobal) {
                    localStorage.setItem('ai-mem-context', JSON.stringify(payload));
                    localStorage.removeItem(keys.config);
                } else {
                    localStorage.setItem(keys.config, JSON.stringify(payload));
                }
                localStorage.setItem(keys.useGlobal, String(!!config.useGlobal));
            }

            async function loadContextConfig() {
                await loadContextDefaults();
                const projectValue = getCurrentProjectValue();
                const config = getContextConfig(projectValue);
                applyContextConfig(config);
            }

            function updateContextConfig() {
                const projectValue = getCurrentProjectValue();
                const config = readContextForm();
                persistContextConfig(config, projectValue);
            }

            function clearTimelineAnchorStorage(projectValue) {
                const keys = getTimelineKeys(projectValue);
                localStorage.removeItem(keys.id);
                localStorage.removeItem(keys.query);
            }

            function clearQueryStorage(projectValue) {
                const key = getQueryKey(projectValue);
                localStorage.removeItem(key);
            }

            function clearQueryUseGlobalStorage(projectValue) {
                const key = getQueryUseGlobalKey(projectValue);
                localStorage.removeItem(key);
            }

            function persistQuery(value) {
                const projectValue = getCurrentProjectValue();
                const key = getQueryKey(projectValue);
                const useGlobalToggle = document.getElementById('queryGlobal');
                const useGlobal = useGlobalToggle ? useGlobalToggle.checked : false;
                localStorage.setItem(getQueryUseGlobalKey(projectValue), String(useGlobal));
                if (!useGlobal) {
                    localStorage.setItem(key, value);
                }
                localStorage.setItem('ai-mem-query', value);
            }

            function updateQueryGlobal() {
                const toggle = document.getElementById('queryGlobal');
                if (!toggle) return;
                const projectValue = getCurrentProjectValue();
                localStorage.setItem(getQueryUseGlobalKey(projectValue), String(toggle.checked));
                loadQuery();
                persistQuery(document.getElementById('query').value || '');
                updateQueryClearButton();
                if (lastMode === 'timeline') {
                    timeline();
                } else {
                    search();
                }
            }

            function persistLastMode(mode) {
                const projectValue = getCurrentProjectValue();
                const key = getModeKey(projectValue);
                localStorage.setItem(key, mode);
                localStorage.setItem('ai-mem-last-mode', mode);
            }

            function loadLastMode() {
                const projectValue = getCurrentProjectValue();
                const key = getModeKey(projectValue);
                let stored = localStorage.getItem(key);
                if (stored === null) {
                    stored = localStorage.getItem('ai-mem-last-mode');
                    if (stored) {
                        localStorage.setItem(key, stored);
                    }
                }
                lastMode = stored || 'search';
            }

            function persistTimelineAnchor() {
                const projectValue = getCurrentProjectValue();
                const keys = getTimelineKeys(projectValue);
                if (timelineAnchorId) {
                    localStorage.setItem(keys.id, timelineAnchorId);
                    localStorage.removeItem(keys.query);
                    return;
                }
                if (timelineQuery) {
                    localStorage.setItem(keys.query, timelineQuery);
                    localStorage.removeItem(keys.id);
                    return;
                }
                localStorage.removeItem(keys.id);
                localStorage.removeItem(keys.query);
            }

            function getAnchorSummary() {
                if (timelineAnchorId) {
                    return { label: 'Anchor', value: timelineAnchorId };
                }
                if (timelineQuery) {
                    return { label: 'Anchor query', value: timelineQuery };
                }
                return null;
            }

            function updateAnchorPill() {
                const pill = document.getElementById('anchorPill');
                const label = document.getElementById('anchorLabel');
                const badge = document.getElementById('anchorBadge');
                const summary = getAnchorSummary();
                if (!pill || !label) return;
                if (summary) {
                    pill.style.display = 'inline-flex';
                    label.textContent = `${summary.label}: ${summary.value}`;
                } else {
                    pill.style.display = 'none';
                }
                if (badge) {
                    if (summary) {
                        badge.style.display = 'inline-flex';
                        badge.innerHTML = '';
                        const labelEl = document.createElement('strong');
                        labelEl.textContent = summary.label;
                        const valueEl = document.createElement('span');
                        valueEl.className = 'anchor-value';
                        valueEl.textContent = summary.value;
                        valueEl.title = summary.value;
                        const clearBtn = document.createElement('button');
                        clearBtn.type = 'button';
                        clearBtn.title = 'Clear anchor';
                        clearBtn.textContent = 'x';
                        clearBtn.addEventListener('click', clearTimelineAnchor);
                        badge.appendChild(labelEl);
                        badge.appendChild(valueEl);
                        badge.appendChild(clearBtn);
                    } else {
                        badge.style.display = 'none';
                        badge.innerHTML = '';
                    }
                }
                updateResultsHeader();
                updateFiltersPill();
                updateTimelineStatus();
            }

            function updateResultsHeader() {
                const header = document.getElementById('resultsHeader');
                if (!header) return;
                const anchorSummary = getAnchorSummary();
                const details = getFilterDetails();
                const filtersLabel = details.count > 0 ? `Filters (${details.count})` : '';
                const icons = details.count ? `<span class="filter-icons">${details.icons}</span>` : '';
                const autoRefresh = document.getElementById('autoRefresh');
                const live = autoRefresh ? autoRefresh.checked : false;
                const modeInput = document.querySelector('input[name="refreshMode"]:checked');
                const modeValue = modeInput ? modeInput.value : 'all';
                const intervalInput = document.getElementById('refreshInterval');
                const interval = Math.max(5, parseInt(intervalInput ? intervalInput.value : '30', 10));
                const compact = window.innerWidth <= 960;
                const pulseToggle = document.getElementById('pulseToggle');
                const pulseOn = pulseToggle ? pulseToggle.checked : true;
                header.classList.toggle('live', live);
                const liveDot = `<span class="live-dot" title="${getLiveTitle(live)}"></span>`;
                const pulseStatus = live && !pulseOn
                    ? '<span class="pulse-status">Pulse disabled</span>'
                    : '';
                let timelineBadge = '';
                if (lastMode === 'timeline' || anchorSummary) {
                    const compact = window.innerWidth <= 960;
                    const suffix = anchorSummary ? '' : (compact ? ' (no)' : ' (no anchor)');
                    const title = anchorSummary ? 'Timeline mode' : 'Timeline mode without an anchor';
                    timelineBadge = `<span class="timeline-badge" title="${title}">Timeline${suffix}</span>`;
                }
                const globalToggle = document.getElementById('autoGlobal');
                const autoScope = globalToggle && globalToggle.checked ? 'global' : 'project';
                const fullModeLabel = modeValue === 'stats' ? 'Stats only' : 'Stats + results';
                const compactModeLabel = modeValue === 'stats' ? 'Stats' : 'All';
                const modeLabel = compact ? compactModeLabel : fullModeLabel;
                const autoScopeMarker = autoScope === 'global' && !compact ? '● ' : '';
                const autoBadge = live
                    ? `<span class="auto-mode-badge" title="Auto-refresh: ${fullModeLabel} • ${interval}s • ${autoScope}">${autoScopeMarker}${modeLabel}</span>`
                    : '';
                if (lastMode === 'timeline') {
                    const depthBefore = timelineDepthBefore || '3';
                    const depthAfter = timelineDepthAfter || '3';
                    let label = `Timeline (${depthBefore} before / ${depthAfter} after)`;
                    if (anchorSummary) {
                        label = `${label} • ${anchorSummary.label}: ${anchorSummary.value}`;
                    }
                    const filterText = filtersLabel ? ` • ${filtersLabel}` : '';
                    header.innerHTML = `<span>${label}${filterText}${icons}${liveDot}${pulseStatus}${timelineBadge}${autoBadge}</span><div class="row inline"><button class="secondary" onclick="clearTimelineAnchor()">Clear anchor</button><button class="secondary" onclick="clearAllFilters()">Clear filters</button><button class="secondary" onclick="search()">Exit</button></div>`;
                    header.style.display = 'flex';
                    return;
                }
                if (lastMode === 'search' && (details.count > 0 || live)) {
                    const query = (document.getElementById('query').value || '').trim();
                    const label = query ? `Search results for "${query}"` : 'Search results';
                    const filterText = filtersLabel ? ` • ${filtersLabel}` : '';
                    header.innerHTML = `<span>${label}${filterText}${icons}${liveDot}${pulseStatus}${autoBadge}</span><div class="row inline"><button class="secondary" onclick="clearAllFilters()">Clear filters</button></div>`;
                    header.style.display = 'flex';
                    return;
                }
                header.style.display = 'none';
                header.innerHTML = '';
            }

            function updateTimelineStatus() {
                const status = document.getElementById('timelineStatus');
                if (!status) return;
                if (lastMode !== 'timeline') {
                    status.style.display = 'none';
                    status.innerHTML = '';
                    return;
                }
                const summary = getAnchorSummary();
                const label = summary ? summary.label : 'Timeline';
                const value = summary ? summary.value : '';
                const valueSpan = value ? `<span class="timeline-value" title="${value}">${value}</span>` : '';
                status.style.display = 'inline-flex';
                status.innerHTML = `
                    <strong>${label}</strong>
                    ${valueSpan}
                    <button class="secondary" type="button" onclick="search()">Exit</button>
                `;
            }

            function getFilterDetails() {
                const dots = [];
                const query = (document.getElementById('query').value || '').trim();
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const type = document.getElementById('type').value;
                const tags = (document.getElementById('tagsFilter').value || '').trim();
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                if (sessionId) {
                    dots.push({ label: 'session', value: sessionId });
                } else if (project) {
                    dots.push({ label: 'project', value: project });
                }
                if (type) dots.push({ label: 'type', value: type });
                if (tags) dots.push({ label: 'tag', value: tags });
                if (dateStart || dateEnd) {
                    dots.push({ label: 'date', value: `${dateStart || '…'} → ${dateEnd || '…'}` });
                }
                if (query) dots.push({ label: 'query', value: query });
                const iconMap = {
                    project: 'P',
                    session: 'S',
                    type: 'T',
                    tag: '#',
                    date: 'D',
                    query: 'Q',
                };
                const icons = dots
                    .map(item => `<span class="filter-icon ${item.label}" title="${item.label}: ${item.value}">${iconMap[item.label] || '?'}</span>`)
                    .join('');
                const summary = dots.map(item => `${item.label}: ${item.value}`).join(' • ');
                return { count: dots.length, icons, summary };
            }

            function updateFiltersPill() {
                const pill = document.getElementById('filtersPill');
                if (!pill) return;
                const details = getFilterDetails();
                updateStatsTitle(details);
                updateTimelineStatus();
                const autoRefresh = document.getElementById('autoRefresh');
                const live = autoRefresh ? autoRefresh.checked : false;
                const pulseToggle = document.getElementById('pulseToggle');
                const pulseOn = pulseToggle ? pulseToggle.checked : true;
                const pills = [
                    pill,
                    document.getElementById('filtersPillSidebar'),
                    document.getElementById('filtersPillStats'),
                ].filter(Boolean);
                pills.forEach(target => {
                    target.style.display = details.count > 0 ? 'inline-flex' : 'none';
                    target.classList.toggle('live', live);
                    if (!details.count) {
                        target.innerHTML = '';
                        target.classList.remove('live');
                        return;
                    }
                    const liveDot = `<span class="live-dot" title="${getLiveTitle(live)}"></span>`;
                    const pulseStatus = live && !pulseOn
                        ? '<span class="pulse-status">Pulse disabled</span>'
                        : '';
                    target.innerHTML = `${details.icons}<span title="${details.summary}">Filters (${details.count})</span>${liveDot}${pulseStatus}<button class="secondary" onclick="clearAllFilters()">Clear</button>`;
                });
            }

            function getLiveTitle(live) {
                if (!live) return 'Auto-refresh off';
                const pulseToggle = document.getElementById('pulseToggle');
                const pulseOn = pulseToggle ? pulseToggle.checked : true;
                return pulseOn ? 'Auto-refresh active' : 'Auto-refresh active (pulse off)';
            }

            function updateStatsTitle(details) {
                const statsTitle = document.getElementById('statsTitle');
                if (!statsTitle) return;
                const current = details || getFilterDetails();
                const count = current.count || 0;
                const autoRefresh = document.getElementById('autoRefresh');
                const live = autoRefresh ? autoRefresh.checked : false;
                const label = count ? `Stats (${count})` : 'Stats';
                const liveDot = document.getElementById('statsLiveDot');
                statsTitle.classList.toggle('live', live);
                statsTitle.childNodes[0].nodeValue = label;
                if (liveDot) {
                    liveDot.title = getLiveTitle(live);
                }
            }

            function updateSidebarLiveBadge() {
                const badge = document.getElementById('sidebarLiveBadge');
                if (!badge) return;
                const autoRefresh = document.getElementById('autoRefresh');
                const live = autoRefresh ? autoRefresh.checked : false;
                const pulseToggle = document.getElementById('pulseToggle');
                const pulseOn = pulseToggle ? pulseToggle.checked : true;
                const compact = window.innerWidth <= 960;
                badge.style.display = live ? 'inline-flex' : 'none';
                badge.classList.toggle('off', live && !pulseOn);
                if (!live) {
                    badge.textContent = '';
                    return;
                }
                if (pulseOn) {
                    badge.textContent = 'Live';
                    return;
                }
                badge.textContent = compact ? 'Live off' : 'Live (pulse off)';
            }

            function clearTimelineAnchor() {
                const projectValue = getCurrentProjectValue();
                timelineAnchorId = '';
                timelineQuery = '';
                clearTimelineAnchorStorage(projectValue);
                updateAnchorPill();
                updateFiltersPill();
                if (lastMode === 'timeline') {
                    timeline();
                }
            }

            async function copyTimelineAnchor() {
                const value = timelineAnchorId || timelineQuery;
                if (!value) {
                    alert('No anchor to copy.');
                    return;
                }
                try {
                    await navigator.clipboard.writeText(value);
                } catch (error) {
                    alert('Copy failed');
                }
            }

            function noteTyping() {
                typingPauseUntil = Date.now() + 1500;
            }

            function shouldPauseAutoRefresh() {
                if (!document.getElementById('autoRefresh').checked) return false;
                if (Date.now() < typingPauseUntil) return true;
                const active = document.activeElement;
                if (active && active.id === 'query') return true;
                return false;
            }

            function setLastDays(days) {
                const end = new Date();
                const start = new Date();
                start.setDate(end.getDate() - (days - 1));
                document.getElementById('dateEnd').value = formatLocalDate(end);
                document.getElementById('dateStart').value = formatLocalDate(start);
                persistFilters();
                loadStats();
                loadTagManager();
                search();
            }

            function setTimelineDepth(before, after) {
                document.getElementById('depthBefore').value = String(before);
                document.getElementById('depthAfter').value = String(after);
                timelineDepthBefore = String(before);
                timelineDepthAfter = String(after);
                localStorage.setItem('ai-mem-timeline-depth-before', timelineDepthBefore);
                localStorage.setItem('ai-mem-timeline-depth-after', timelineDepthAfter);
                updateResultsHeader();
                if (lastMode === 'timeline') {
                    timeline();
                }
            }

            function setListLimit(value) {
                listLimit = String(value);
                document.getElementById('limit').value = listLimit;
                localStorage.setItem('ai-mem-list-limit', listLimit);
                updateLimitWarning();
            }

            function clearDates() {
                document.getElementById('dateStart').value = '';
                document.getElementById('dateEnd').value = '';
                persistFilters();
                loadStats();
                loadTagManager();
                search();
            }

            function clearQuery() {
                const input = document.getElementById('query');
                if (!input) return;
                input.value = '';
                timelineAnchorId = '';
                timelineQuery = '';
                persistQuery('');
                persistTimelineAnchor();
                updateQueryClearButton();
                updateAnchorPill();
                updateResultsHeader();
                updateFiltersPill();
            }

            function clearAllFilters() {
                if (!confirm('Clear all filters and reset the view?')) {
                    return;
                }
                const previousProject = getCurrentProjectValue();
                document.getElementById('query').value = '';
                document.getElementById('project').value = '';
                document.getElementById('sessionId').value = '';
                document.getElementById('type').value = '';
                document.getElementById('tagsFilter').value = '';
                document.getElementById('dateStart').value = '';
                document.getElementById('dateEnd').value = '';
                timelineAnchorId = '';
                timelineQuery = '';
                selectedProject = '';
                selectedSessionId = '';
                selectedType = '';
                selectedTags = '';
                selectedDateStart = '';
                selectedDateEnd = '';
                listLimit = '10';
                timelineDepthBefore = '3';
                timelineDepthAfter = '3';
                document.getElementById('limit').value = listLimit;
                document.getElementById('depthBefore').value = timelineDepthBefore;
                document.getElementById('depthAfter').value = timelineDepthAfter;
                localStorage.removeItem('ai-mem-selected-project');
                localStorage.removeItem('ai-mem-selected-session-id');
                localStorage.removeItem('ai-mem-selected-type');
                localStorage.removeItem('ai-mem-selected-tags');
                localStorage.removeItem('ai-mem-date-start');
                localStorage.removeItem('ai-mem-date-end');
                clearQueryStorage(previousProject);
                clearQueryStorage('');
                clearQueryUseGlobalStorage(previousProject);
                clearQueryUseGlobalStorage('');
                localStorage.removeItem('ai-mem-query');
                clearTimelineAnchorStorage(previousProject);
                clearTimelineAnchorStorage('');
                localStorage.removeItem('ai-mem-timeline-anchor-id');
                localStorage.removeItem('ai-mem-timeline-anchor-query');
                localStorage.removeItem('ai-mem-list-limit');
                localStorage.removeItem('ai-mem-timeline-depth-before');
                localStorage.removeItem('ai-mem-timeline-depth-after');
                updateLimitWarning();
                updateQueryClearButton();
                updateAnchorPill();
                updateResultsHeader();
                updateFiltersPill();
                loadStats();
                loadTagManager();
                search();
                restartStreamIfActive();
            }

            function applyTagFilter(tag, append = false) {
                if (!tag) return;
                const input = document.getElementById('tagsFilter');
                if (!input) return;
                const current = (input.value || '')
                    .split(',')
                    .map(item => item.trim())
                    .filter(item => item);
                let next = [];
                if (append) {
                    const set = new Set(current);
                    set.add(tag);
                    next = Array.from(set);
                } else {
                    next = [tag];
                }
                input.value = next.join(', ');
                persistFilters();
                loadStats();
                if (lastMode === 'timeline') {
                    timeline({ useInput: false });
                } else {
                    search();
                }
            }

            function bindTagPills(container) {
                if (!container) return;
                container.querySelectorAll('.pill.clickable').forEach(pill => {
                    if (pill.dataset.bound === 'true') return;
                    pill.dataset.bound = 'true';
                    pill.addEventListener('click', event => {
                        const value = pill.dataset.tag || '';
                        if (!value) return;
                        const tag = decodeURIComponent(value);
                        applyTagFilter(tag, event.shiftKey);
                        const tagOld = document.getElementById('tagOld');
                        if (tagOld) {
                            tagOld.value = tag;
                        }
                    });
                });
            }

            function persistFilters() {
                selectedSessionId = (document.getElementById('sessionId')?.value || '').trim();
                selectedType = document.getElementById('type').value || '';
                selectedTags = (document.getElementById('tagsFilter').value || '').trim();
                selectedDateStart = document.getElementById('dateStart').value || '';
                selectedDateEnd = document.getElementById('dateEnd').value || '';
                localStorage.setItem('ai-mem-selected-session-id', selectedSessionId);
                localStorage.setItem('ai-mem-selected-type', selectedType);
                localStorage.setItem('ai-mem-selected-tags', selectedTags);
                localStorage.setItem('ai-mem-date-start', selectedDateStart);
                localStorage.setItem('ai-mem-date-end', selectedDateEnd);
            }

            function loadSavedFilters() {
                const stored = localStorage.getItem('ai-mem-saved-filters');
                if (!stored) {
                    savedFilters = [];
                    renderSavedFilters();
                    return;
                }
                try {
                    const parsed = JSON.parse(stored);
                    savedFilters = Array.isArray(parsed) ? parsed : [];
                } catch (error) {
                    savedFilters = [];
                }
                renderSavedFilters();
            }

            function persistSavedFilters() {
                localStorage.setItem('ai-mem-saved-filters', JSON.stringify(savedFilters));
                renderSavedFilters();
            }

            function renderSavedFilters() {
                const select = document.getElementById('savedFilters');
                if (!select) return;
                select.innerHTML = '<option value="">Select saved filter</option>';
                savedFilters.forEach(item => {
                    const option = document.createElement('option');
                    option.value = item.name || '';
                    option.textContent = item.name || '';
                    select.appendChild(option);
                });
            }

            function collectFilterState() {
                return {
                    name: '',
                    mode: lastMode || 'search',
                    query: (document.getElementById('query')?.value || '').trim(),
                    project: document.getElementById('project').value || '',
                    sessionId: (document.getElementById('sessionId')?.value || '').trim(),
                    type: document.getElementById('type').value || '',
                    tags: (document.getElementById('tagsFilter')?.value || '').trim(),
                    dateStart: document.getElementById('dateStart')?.value || '',
                    dateEnd: document.getElementById('dateEnd')?.value || '',
                    limit: document.getElementById('limit')?.value || listLimit,
                    depthBefore: document.getElementById('depthBefore')?.value || timelineDepthBefore,
                    depthAfter: document.getElementById('depthAfter')?.value || timelineDepthAfter,
                    anchorId: timelineAnchorId || '',
                };
            }

            async function applyFilterState(state) {
                if (!state) return;
                const projectSelect = document.getElementById('project');
                if (projectSelect) {
                    projectSelect.value = state.project || '';
                }
                document.getElementById('query').value = state.query || '';
                document.getElementById('sessionId').value = state.sessionId || '';
                document.getElementById('type').value = state.type || '';
                document.getElementById('tagsFilter').value = state.tags || '';
                document.getElementById('dateStart').value = state.dateStart || '';
                document.getElementById('dateEnd').value = state.dateEnd || '';
                document.getElementById('limit').value = state.limit || listLimit;
                document.getElementById('depthBefore').value = state.depthBefore || timelineDepthBefore;
                document.getElementById('depthAfter').value = state.depthAfter || timelineDepthAfter;

                selectedProject = state.project || '';
                localStorage.setItem('ai-mem-selected-project', selectedProject);
                listLimit = state.limit || listLimit;
                timelineDepthBefore = state.depthBefore || timelineDepthBefore;
                timelineDepthAfter = state.depthAfter || timelineDepthAfter;
                localStorage.setItem('ai-mem-list-limit', listLimit);
                localStorage.setItem('ai-mem-timeline-depth-before', timelineDepthBefore);
                localStorage.setItem('ai-mem-timeline-depth-after', timelineDepthAfter);

                timelineAnchorId = state.anchorId || '';
                timelineQuery = state.query || '';
                persistFilters();
                persistQuery(state.query || '');
                persistTimelineAnchor();
                updateQueryClearButton();
                updateAnchorPill();
                updateResultsHeader();
                updateFiltersPill();
                await loadStats();
                await loadTagManager();
                if (state.mode === 'timeline') {
                    lastMode = 'timeline';
                    persistLastMode('timeline');
                    await timeline();
                } else {
                    lastMode = 'search';
                    persistLastMode('search');
                    await search();
                }
                restartStreamIfActive();
            }

            function saveCurrentFilter() {
                const nameInput = document.getElementById('savedFilterName');
                const name = (nameInput?.value || '').trim();
                if (!name) {
                    alert('Enter a name for this filter.');
                    return;
                }
                const state = collectFilterState();
                state.name = name;
                const existingIndex = savedFilters.findIndex(item => item.name === name);
                if (existingIndex >= 0) {
                    savedFilters[existingIndex] = state;
                } else {
                    savedFilters.unshift(state);
                }
                persistSavedFilters();
                if (nameInput) {
                    nameInput.value = '';
                }
            }

            async function applySavedFilter() {
                const select = document.getElementById('savedFilters');
                const name = select?.value || '';
                if (!name) return;
                const state = savedFilters.find(item => item.name === name);
                await applyFilterState(state);
            }

            function deleteSavedFilter() {
                const select = document.getElementById('savedFilters');
                const name = select?.value || '';
                if (!name) return;
                savedFilters = savedFilters.filter(item => item.name !== name);
                persistSavedFilters();
                if (select) {
                    select.value = '';
                }
            }

            function buildTagScopeParams() {
                const params = new URLSearchParams();
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const type = document.getElementById('type').value;
                const tagsFilter = (document.getElementById('tagsFilter')?.value || '').trim();
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                if (sessionId) {
                    params.append('session_id', sessionId);
                } else if (project) {
                    params.append('project', project);
                }
                if (type) params.append('obs_type', type);
                if (tagsFilter) params.append('tags', tagsFilter);
                if (dateStart) params.append('date_start', dateStart);
                if (dateEnd) params.append('date_end', dateEnd);
                params.append('limit', '50');
                return params;
            }

            function buildTagScopePayload() {
                const payload = {};
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const type = document.getElementById('type').value;
                const tagsFilter = (document.getElementById('tagsFilter')?.value || '').trim();
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                if (sessionId) {
                    payload.session_id = sessionId;
                } else if (project) {
                    payload.project = project;
                }
                if (type) payload.obs_type = type;
                if (tagsFilter) payload.tags = tagsFilter;
                if (dateStart) payload.date_start = dateStart;
                if (dateEnd) payload.date_end = dateEnd;
                return payload;
            }

            async function loadTagManager() {
                const params = buildTagScopeParams();
                const response = await fetch(`/api/tags?${params.toString()}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                const data = response.ok ? await response.json() : [];
                const list = document.getElementById('tagList');
                const meta = document.getElementById('tagMeta');
                if (!list || !meta) return;
                const scopeParts = [];
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const project = document.getElementById('project').value;
                const type = document.getElementById('type').value;
                const tagsFilter = (document.getElementById('tagsFilter')?.value || '').trim();
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                if (sessionId) {
                    scopeParts.push(`session ${sessionId}`);
                } else if (project) {
                    scopeParts.push(project);
                } else {
                    scopeParts.push('all projects');
                }
                if (type) scopeParts.push(type);
                if (tagsFilter) scopeParts.push(`tags: ${tagsFilter}`);
                if (dateStart || dateEnd) {
                    scopeParts.push(`${dateStart || '...'} → ${dateEnd || '...'}`);
                }
                meta.textContent = `Scope: ${scopeParts.join(' • ')}`;
                if (!data || !data.length) {
                    list.innerHTML = '<div class="subtitle">No tags found.</div>';
                    return;
                }
                list.innerHTML = data.map(item => {
                    const tag = item.tag || '-';
                    const encoded = encodeURIComponent(tag);
                    return `
                        <div class="tag-item">
                            <span class="pill clickable" data-tag="${encoded}" title="Filter by tag (shift+click to add)">${escapeHtml(tag)}</span>
                            <span class="tag-count">${item.count || 0}</span>
                        </div>
                    `;
                }).join('');
                bindTagPills(list);
            }

            async function addTag() {
                const addInput = document.getElementById('tagAdd');
                const tag = (addInput?.value || '').trim();
                if (!tag) {
                    alert('Provide a tag to add.');
                    return;
                }
                const payload = buildTagScopePayload();
                payload.tag = tag;
                const response = await fetch('/api/tags/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                    body: JSON.stringify(payload),
                });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    alert('Failed to add tag');
                    return;
                }
                if (addInput) addInput.value = '';
                await loadTagManager();
                await loadStats();
                if (lastMode === 'timeline') {
                    timeline({ useInput: false });
                } else {
                    search();
                }
            }

            async function renameTag() {
                const oldInput = document.getElementById('tagOld');
                const newInput = document.getElementById('tagNew');
                const oldTag = (oldInput?.value || '').trim();
                const newTag = (newInput?.value || '').trim();
                if (!oldTag || !newTag) {
                    alert('Provide both a tag and a new value.');
                    return;
                }
                if (!confirm(`Rename tag "${oldTag}" to "${newTag}"?`)) return;
                const payload = buildTagScopePayload();
                payload.old_tag = oldTag;
                payload.new_tag = newTag;
                const response = await fetch('/api/tags/rename', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                    body: JSON.stringify(payload),
                });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    alert('Failed to rename tag');
                    return;
                }
                if (newInput) newInput.value = '';
                await loadTagManager();
                await loadStats();
                if (lastMode === 'timeline') {
                    timeline({ useInput: false });
                } else {
                    search();
                }
            }

            async function deleteTag() {
                const oldInput = document.getElementById('tagOld');
                const tag = (oldInput?.value || '').trim();
                if (!tag) {
                    alert('Provide a tag to delete.');
                    return;
                }
                if (!confirm(`Remove tag "${tag}" from matching observations?`)) return;
                const payload = buildTagScopePayload();
                payload.tag = tag;
                const response = await fetch('/api/tags/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                    body: JSON.stringify(payload),
                });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    alert('Failed to delete tag');
                    return;
                }
                await loadTagManager();
                await loadStats();
                if (lastMode === 'timeline') {
                    timeline({ useInput: false });
                } else {
                    search();
                }
            }

            function updateQueryClearButton() {
                const input = document.getElementById('query');
                const button = document.getElementById('queryClear');
                if (!input || !button) return;
                const hasValue = Boolean(input.value.trim());
                button.disabled = !hasValue;
                button.title = hasValue ? 'Clear query' : 'Query is empty';
            }

            function updateLimitWarning() {
                const limitValue = parseInt(document.getElementById('limit').value || '0', 10);
                const warning = document.getElementById('limitWarning');
                if (!warning) return;
                warning.style.display = limitValue > 100 ? 'block' : 'none';
            }

            async function refreshResultsForFilters() {
                if (lastMode === 'timeline') {
                    const query = (document.getElementById('query').value || '').trim();
                    if (timelineAnchorId || timelineQuery || query) {
                        await timeline();
                        return;
                    }
                }
                await search();
            }

            async function refreshAll() {
                if (shouldPauseAutoRefresh()) {
                    return;
                }
                await loadStats();
                const mode = localStorage.getItem('ai-mem-auto-refresh-mode') || 'all';
                if (mode === 'stats') {
                    return;
                }
                if (lastMode === 'timeline') {
                    await timeline({ useInput: false });
                    return;
                }
                await search();
            }

            function updateAutoRefresh() {
                const enabled = document.getElementById('autoRefresh').checked;
                const intervalInput = document.getElementById('refreshInterval');
                const interval = Math.max(5, parseInt(intervalInput.value || '30', 10));
                intervalInput.value = String(interval);
                const modeInput = document.querySelector('input[name="refreshMode"]:checked');
                const modeValue = modeInput ? modeInput.value : 'all';
                const projectValue = getCurrentProjectValue();
                const keys = getAutoRefreshKeys(projectValue);
                const globalToggle = document.getElementById('autoGlobal');
                const useGlobal = globalToggle ? globalToggle.checked : false;
                localStorage.setItem(keys.useGlobal, String(useGlobal));
                if (useGlobal) {
                    localStorage.setItem('ai-mem-auto-refresh', String(enabled));
                    localStorage.setItem('ai-mem-auto-refresh-interval', String(interval));
                    localStorage.setItem('ai-mem-auto-refresh-mode', modeValue);
                } else {
                    localStorage.setItem(keys.enabled, String(enabled));
                    localStorage.setItem(keys.interval, String(interval));
                    localStorage.setItem(keys.mode, modeValue);
                }
                localStorage.setItem('ai-mem-auto-refresh', String(enabled));
                localStorage.setItem('ai-mem-auto-refresh-interval', String(interval));
                localStorage.setItem('ai-mem-auto-refresh-mode', modeValue);
                const indicator = document.getElementById('liveIndicator');
                if (indicator) {
                    indicator.classList.toggle('active', enabled);
                }
                updateAutoModeLabel(modeValue, enabled);
                updateResultsHeader();
                updateStatsTitle();
                if (autoRefreshTimer) {
                    clearInterval(autoRefreshTimer);
                    autoRefreshTimer = null;
                }
                if (enabled) {
                    refreshAll();
                    autoRefreshTimer = setInterval(refreshAll, interval * 1000);
                }
                updateSidebarLiveBadge();
            }

            function updateAutoModeLabel(modeValue, enabled) {
                const label = document.getElementById('autoModeLabel');
                if (!label) return;
                if (!enabled) {
                    label.textContent = '';
                    label.title = '';
                    label.classList.remove('global');
                    label.style.display = 'none';
                    return;
                }
                const modeText = modeValue === 'stats' ? 'Stats only' : 'Stats + results';
                const intervalInput = document.getElementById('refreshInterval');
                const interval = Math.max(5, parseInt(intervalInput ? intervalInput.value : '30', 10));
                label.textContent = `Mode: ${modeText} • ${interval}s`;
                const globalToggle = document.getElementById('autoGlobal');
                const scope = globalToggle && globalToggle.checked ? 'global' : 'project';
                label.title = `Auto-refresh: ${modeText} • ${interval}s • ${scope}`;
                label.classList.toggle('global', scope === 'global');
                label.style.display = 'inline-flex';
            }

            function updateAutoGlobal() {
                const toggle = document.getElementById('autoGlobal');
                if (!toggle) return;
                const projectValue = getCurrentProjectValue();
                const keys = getAutoRefreshKeys(projectValue);
                localStorage.setItem(keys.useGlobal, String(toggle.checked));
                loadAutoRefresh();
            }

            function getPulseConfig(project) {
                const prefersReduced = window.matchMedia
                    ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
                    : false;
                const globalStored = localStorage.getItem('ai-mem-pulse-default');
                const globalEnabled = globalStored === null ? !prefersReduced : globalStored === 'true';
                const projectKey = `ai-mem-pulse-${project}`;
                const projectStored = localStorage.getItem(projectKey);
                const useGlobalKey = `ai-mem-pulse-use-global-${project}`;
                const useGlobalStored = localStorage.getItem(useGlobalKey);
                const useGlobal = useGlobalStored === null
                    ? projectStored === null
                    : useGlobalStored === 'true';
                const enabled = useGlobal
                    ? globalEnabled
                    : (projectStored === null ? globalEnabled : projectStored === 'true');
                return { enabled, useGlobal };
            }

            function updatePulseToggle() {
                const toggle = document.getElementById('pulseToggle');
                if (!toggle) return;
                const enabled = toggle.checked;
                const project = document.getElementById('project').value || 'all';
                const useGlobalToggle = document.getElementById('pulseGlobal');
                const useGlobal = useGlobalToggle ? useGlobalToggle.checked : false;
                if (useGlobal) {
                    localStorage.setItem('ai-mem-pulse-default', String(enabled));
                    localStorage.setItem(`ai-mem-pulse-use-global-${project}`, 'true');
                } else {
                    localStorage.setItem(`ai-mem-pulse-${project}`, String(enabled));
                    if (useGlobalToggle) {
                        localStorage.setItem(`ai-mem-pulse-use-global-${project}`, 'false');
                    }
                }
                document.body.classList.toggle('pulse-off', !enabled);
                updateResultsHeader();
                updateFiltersPill();
                updateStatsTitle();
                updateSidebarLiveBadge();
            }

            function updatePulseGlobal() {
                const toggle = document.getElementById('pulseToggle');
                const globalToggle = document.getElementById('pulseGlobal');
                if (!toggle || !globalToggle) return;
                const project = document.getElementById('project').value || 'all';
                const useGlobal = globalToggle.checked;
                localStorage.setItem(`ai-mem-pulse-use-global-${project}`, String(useGlobal));
                if (!useGlobal) {
                    const projectKey = `ai-mem-pulse-${project}`;
                    if (localStorage.getItem(projectKey) === null) {
                        localStorage.setItem(projectKey, String(toggle.checked));
                    }
                }
                loadPulseToggle();
            }

            function loadPulseToggle() {
                const toggle = document.getElementById('pulseToggle');
                if (!toggle) return;
                const project = document.getElementById('project').value || 'all';
                const config = getPulseConfig(project);
                toggle.checked = config.enabled;
                const globalToggle = document.getElementById('pulseGlobal');
                if (globalToggle) {
                    globalToggle.checked = config.useGlobal;
                }
                document.body.classList.toggle('pulse-off', !config.enabled);
                updateResultsHeader();
                updateFiltersPill();
                updateStatsTitle();
                updateSidebarLiveBadge();
            }

            function loadAutoRefresh() {
                const projectValue = getCurrentProjectValue();
                const keys = getAutoRefreshKeys(projectValue);
                const globalEnabled = localStorage.getItem('ai-mem-auto-refresh');
                const globalInterval = localStorage.getItem('ai-mem-auto-refresh-interval');
                const globalMode = localStorage.getItem('ai-mem-auto-refresh-mode');
                let enabledValue = localStorage.getItem(keys.enabled);
                let intervalValue = localStorage.getItem(keys.interval);
                let modeValue = localStorage.getItem(keys.mode);
                const useGlobalStored = localStorage.getItem(keys.useGlobal);
                const useGlobal = useGlobalStored === null
                    ? enabledValue === null && intervalValue === null && modeValue === null
                    : useGlobalStored === 'true';
                if (useGlobal) {
                    enabledValue = globalEnabled ?? enabledValue;
                    intervalValue = globalInterval ?? intervalValue;
                    modeValue = globalMode ?? modeValue;
                } else {
                    if (enabledValue === null && globalEnabled !== null) {
                        enabledValue = globalEnabled;
                        localStorage.setItem(keys.enabled, globalEnabled);
                    }
                    if (intervalValue === null && globalInterval !== null) {
                        intervalValue = globalInterval;
                        localStorage.setItem(keys.interval, globalInterval);
                    }
                    if (modeValue === null && globalMode) {
                        modeValue = globalMode;
                        localStorage.setItem(keys.mode, globalMode);
                    }
                }
                const globalToggle = document.getElementById('autoGlobal');
                if (globalToggle) {
                    globalToggle.checked = useGlobal;
                }
                const enabled = enabledValue === 'true';
                const interval = parseInt(intervalValue || '30', 10);
                const mode = modeValue || 'all';
                document.getElementById('autoRefresh').checked = enabled;
                document.getElementById('refreshInterval').value = String(interval);
                const modeInputs = document.querySelectorAll('input[name="refreshMode"]');
                modeInputs.forEach(input => {
                    input.checked = input.value === mode;
                });
                updateAutoRefresh();
            }

            function loadTimelineAnchor() {
                const projectValue = getCurrentProjectValue();
                const keys = getTimelineKeys(projectValue);
                let anchorId = localStorage.getItem(keys.id) || '';
                let anchorQuery = localStorage.getItem(keys.query) || '';
                if (!anchorId && !anchorQuery) {
                    const legacyId = localStorage.getItem('ai-mem-timeline-anchor-id') || '';
                    const legacyQuery = localStorage.getItem('ai-mem-timeline-anchor-query') || '';
                    if (legacyId || legacyQuery) {
                        anchorId = legacyId;
                        anchorQuery = legacyQuery;
                        if (legacyId) {
                            localStorage.setItem(keys.id, legacyId);
                        }
                        if (legacyQuery) {
                            localStorage.setItem(keys.query, legacyQuery);
                        }
                        localStorage.removeItem('ai-mem-timeline-anchor-id');
                        localStorage.removeItem('ai-mem-timeline-anchor-query');
                    }
                }
                timelineAnchorId = anchorId;
                timelineQuery = anchorQuery;
                updateAnchorPill();
            }

            function loadTimelineDepth() {
                timelineDepthBefore = localStorage.getItem('ai-mem-timeline-depth-before') || '3';
                timelineDepthAfter = localStorage.getItem('ai-mem-timeline-depth-after') || '3';
                document.getElementById('depthBefore').value = timelineDepthBefore;
                document.getElementById('depthAfter').value = timelineDepthAfter;
                updateResultsHeader();
            }

            function loadListLimit() {
                listLimit = localStorage.getItem('ai-mem-list-limit') || '10';
                document.getElementById('limit').value = listLimit;
                updateLimitWarning();
            }

            function loadSelectedProject() {
                selectedProject = localStorage.getItem('ai-mem-selected-project') || '';
            }

            function loadSelectedFilters() {
                selectedSessionId = localStorage.getItem('ai-mem-selected-session-id') || '';
                selectedType = localStorage.getItem('ai-mem-selected-type') || '';
                selectedTags = localStorage.getItem('ai-mem-selected-tags') || '';
                selectedDateStart = localStorage.getItem('ai-mem-date-start') || '';
                selectedDateEnd = localStorage.getItem('ai-mem-date-end') || '';
                const sessionInput = document.getElementById('sessionId');
                if (sessionInput) {
                    sessionInput.value = selectedSessionId;
                }
                if (selectedType) {
                    document.getElementById('type').value = selectedType;
                }
                if (selectedTags) {
                    document.getElementById('tagsFilter').value = selectedTags;
                }
                document.getElementById('dateStart').value = selectedDateStart;
                document.getElementById('dateEnd').value = selectedDateEnd;
                restartStreamIfActive();
            }

            function loadQuery() {
                const input = document.getElementById('query');
                if (!input) return;
                const projectValue = getCurrentProjectValue();
                const config = getQueryConfig(projectValue);
                input.value = config.value || '';
                const toggle = document.getElementById('queryGlobal');
                if (toggle) {
                    toggle.checked = config.useGlobal;
                }
                updateQueryClearButton();
                updateFiltersPill();
            }

            function getAuthHeaders() {
                const token = localStorage.getItem('ai-mem-token');
                if (!token) return {};
                return { 'Authorization': `Bearer ${token}` };
            }

            async function loadStats() {
                const params = buildStatsParams({ type_tag_limit: 3 });
                const response = await fetch(`/api/stats?${params.toString()}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                const data = await response.json();
                renderStats(data);
                updateLastUpdate();
            }

            function renderStats(data) {
                const container = document.getElementById('stats');
                if (!data) {
                    container.innerHTML = '<div class="subtitle">No stats available.</div>';
                    return;
                }
                const total = data.total || 0;
                const dayLimit = data.day_limit || 7;
                const trendDelta = data.trend_delta || 0;
                const trendPct = data.trend_pct;
                let trendLabel = '0';
                if (trendPct !== null && trendPct !== undefined) {
                    const sign = trendDelta > 0 ? '+' : '';
                    trendLabel = `${sign}${trendDelta} (${sign}${trendPct.toFixed(1)}%)`;
                } else if (trendDelta !== 0) {
                    const sign = trendDelta > 0 ? '+' : '';
                    trendLabel = `${sign}${trendDelta}`;
                }
                const trendClass = trendDelta > 0 ? 'up' : trendDelta < 0 ? 'down' : 'flat';
                let html = `
                    <div class="stat-row">
                        <span>Total observations</span>
                        <strong>${total}</strong>
                    </div>
                `;
                html += `
                    <div class="stat-row">
                        <span>Recent (last ${dayLimit} days)</span>
                        <strong>${data.recent_total || 0}</strong>
                    </div>
                    <div class="stat-row">
                        <span>Change vs previous</span>
                        <span class="trend ${trendClass}">${trendLabel}</span>
                    </div>
                `;
                html += '<div class="stat-section">By type</div>';
                if (data.by_type && data.by_type.length) {
                    html += data.by_type.map(item => `
                        <div class="stat-row">
                            <span>${item.type}</span>
                            <strong>${item.count}</strong>
                        </div>
                    `).join('');
                } else {
                    html += '<div class="subtitle">No data</div>';
                }

                const project = document.getElementById('project').value;
                if (!project) {
                    html += '<div class="stat-section">By project</div>';
                    if (data.by_project && data.by_project.length) {
                        const topProjects = data.by_project.slice(0, 8);
                        html += topProjects.map(item => `
                            <div class="stat-row">
                                <span>${item.project}</span>
                                <strong>${item.count}</strong>
                            </div>
                        `).join('');
                        const extra = data.by_project.length - topProjects.length;
                        if (extra > 0) {
                            html += `<div class="subtitle">+${extra} more</div>`;
                        }
                        const maxProject = Math.max(...topProjects.map(item => item.count || 0), 1);
                        html += '<div class="project-bars">';
                        html += topProjects.map(item => {
                            const width = Math.max(6, Math.round((item.count / maxProject) * 100));
                            return `
                                <div class="project-bar">
                                    <span title="${item.project}">${item.project}</span>
                                    <div class="bar-track">
                                        <div class="bar-fill" style="width:${width}%"></div>
                                    </div>
                                    <strong>${item.count}</strong>
                                </div>
                            `;
                        }).join('');
                        html += '</div>';
                    } else {
                        html += '<div class="subtitle">No data</div>';
                    }
                }
                const type = document.getElementById('type').value;
                const tagTitle = type ? `Top tags for ${type}` : 'Top tags';
                html += `<div class="stat-section">${tagTitle}</div>`;
                if (data.top_tags && data.top_tags.length) {
                    html += data.top_tags.map(item => `
                        <div class="stat-row">
                            <span>${item.tag}</span>
                            <strong>${item.count}</strong>
                        </div>
                    `).join('');
                } else {
                    html += '<div class="subtitle">No tags</div>';
                }
                html += '<div class="stat-section">Recent days</div>';
                if (data.by_day && data.by_day.length) {
                    html += data.by_day.map(item => `
                        <div class="stat-row">
                            <span>${item.day}</span>
                            <strong>${item.count}</strong>
                        </div>
                    `).join('');
                    const series = [...data.by_day].reverse();
                    const maxCount = Math.max(...series.map(item => item.count || 0), 1);
                    const bars = series.map(item => {
                        const height = Math.max(8, Math.round((item.count / maxCount) * 100));
                        return `<div class="mini-bar" style="height:${height}%;" title="${item.day} • ${item.count}"></div>`;
                    }).join('');
                    html += `<div class="mini-chart" aria-label="Recent activity chart">${bars}</div>`;
                } else {
                    html += '<div class="subtitle">No data</div>';
                }
                if (!type && data.top_tags_by_type && data.top_tags_by_type.length) {
                    html += '<div class="stat-section">Top tags by type</div>';
                    data.top_tags_by_type.forEach(group => {
                        if (!group.tags || !group.tags.length) return;
                        html += `<div class="subtitle">${group.type}</div>`;
                        html += group.tags.map(item => `
                            <div class="stat-row">
                                <span>${item.tag}</span>
                                <strong>${item.count}</strong>
                            </div>
                        `).join('');
                    });
                }
                container.innerHTML = html;
            }

            async function renderResults(data) {
                const container = document.getElementById('results');
                container.innerHTML = '';
                if (!data.length) {
                    container.innerHTML = '<div class="subtitle">No results yet.</div>';
                    return;
                }
                data.forEach(mem => {
                    const div = document.createElement('div');
                    div.className = 'card';
                    const summaryText = mem.summary || '(no summary)';
                    const tokenEstimate = estimateTokens(mem.summary || '');
                    const tokenLabel = tokenEstimate ? ` • ~${tokenEstimate} tok` : '';
                    div.innerHTML = `
                        <div>${summaryText}</div>
                        <div class="meta">
                            ${mem.project || 'Global'} • ${mem.type || 'note'} • ${mem.id}${tokenLabel}
                        </div>
                    `;
                    div.onclick = () => loadDetail(mem.id);
                    container.appendChild(div);
                });
            }

            async function search() {
                lastMode = 'search';
                persistLastMode('search');
                const response = await fetch(`/api/search?${buildQueryParams()}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                const data = await response.json();
                renderResults(data);
                updateResultsHeader();
                updateFiltersPill();
                updateTimelineStatus();
            }

            async function timeline(options = {}) {
                lastMode = 'timeline';
                persistLastMode('timeline');
                const params = new URLSearchParams();
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const tags = (document.getElementById('tagsFilter').value || '').trim();
                const useInput = options.useInput !== false;
                const queryInput = document.getElementById('query');
                const query = useInput && queryInput ? queryInput.value : '';
                const type = document.getElementById('type').value;
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                const depthBefore = document.getElementById('depthBefore').value;
                const depthAfter = document.getElementById('depthAfter').value;
                timelineDepthBefore = depthBefore || timelineDepthBefore;
                timelineDepthAfter = depthAfter || timelineDepthAfter;
                localStorage.setItem('ai-mem-timeline-depth-before', timelineDepthBefore);
                localStorage.setItem('ai-mem-timeline-depth-after', timelineDepthAfter);
                if (sessionId) {
                    params.append('session_id', sessionId);
                } else if (project) {
                    params.append('project', project);
                }
                if (type) params.append('obs_type', type);
                if (tags) params.append('tags', tags);
                if (dateStart) params.append('date_start', dateStart);
                if (dateEnd) params.append('date_end', dateEnd);
                if (timelineAnchorId) {
                    params.append('anchor_id', timelineAnchorId);
                } else {
                    const resolvedQuery = useInput ? (query || timelineQuery) : timelineQuery;
                    if (resolvedQuery) {
                        params.append('query', resolvedQuery);
                        if (useInput) {
                            timelineQuery = resolvedQuery;
                        }
                    }
                }
                if (depthBefore) params.append('depth_before', depthBefore);
                if (depthAfter) params.append('depth_after', depthAfter);
                const response = await fetch(`/api/timeline?${params.toString()}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                const data = await response.json();
                renderResults(data);
                updateAnchorPill();
                updateFiltersPill();
                updateTimelineStatus();
                persistTimelineAnchor();
            }

            async function loadDetail(id) {
                currentObservationId = id;
                const response = await fetch(`/api/observations/${id}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                const mem = await response.json();
                if (lastMode === 'timeline') {
                    timelineAnchorId = id;
                    timelineQuery = '';
                    updateAnchorPill();
                    persistTimelineAnchor();
                }
                const detail = document.getElementById('detail');
                const tags = (mem.tags || []).map(tag => {
                    const encoded = encodeURIComponent(tag);
                    return `<span class="pill clickable" data-tag="${encoded}" title="Filter by tag (shift+click to add)">${escapeHtml(tag)}</span>`;
                }).join('');
                const tagLine = tags || '<span class="subtitle">No tags</span>';
                const sessionInfo = mem.session_id ? ` • session ${mem.session_id}` : '';
                const sessionButton = mem.session_id
                    ? `<button class="secondary" onclick="loadSessionDetail('${mem.session_id}')">View session</button>`
                    : '';
                detail.innerHTML = `
                    <div><strong>${mem.summary || '(no summary)'}</strong></div>
                    <div class="meta">${mem.project || 'Global'} • ${mem.type || 'note'} • ${mem.id}${sessionInfo}</div>
                    <div class="accent">Tags</div>
                    <div>${tagLine}</div>
                    <div class="tag-editor">
                        <input type="text" id="tagsEditor" placeholder="comma,separated tags">
                        <button class="secondary" onclick="saveTags()">Save tags</button>
                    </div>
                    <div class="button-row">
                        <button onclick="copyId()">Copy ID</button>
                        <button class="secondary" onclick="copyCitation()">Copy URL</button>
                        ${sessionButton}
                        <button class="secondary" onclick="deleteObservation()">Delete</button>
                    </div>
                    <div class="accent">Content</div>
                    <pre>${mem.content || ''}</pre>
                `;
                const tagInput = document.getElementById('tagsEditor');
                if (tagInput) {
                    tagInput.value = (mem.tags || []).join(', ');
                }
                bindTagPills(detail);
            }

            async function saveTags() {
                if (!currentObservationId) return;
                const tagsInput = document.getElementById('tagsEditor');
                const tags = (tagsInput?.value || '')
                    .split(',')
                    .map(item => item.trim())
                    .filter(item => item);
                const response = await fetch(`/api/observations/${currentObservationId}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                    body: JSON.stringify({ tags }),
                });
                if (await handleAuthError(response)) return;
                if (!response.ok) {
                    alert('Failed to update tags');
                    return;
                }
                await loadDetail(currentObservationId);
                await loadStats();
                await loadTagManager();
                if (lastMode === 'timeline') {
                    timeline({ useInput: false });
                } else {
                    search();
                }
            }

            async function deleteObservation() {
                if (!currentObservationId) return;
                if (!confirm(`Delete observation ${currentObservationId}?`)) return;
                const response = await fetch(`/api/observations/${currentObservationId}`, {
                    method: 'DELETE',
                    headers: getAuthHeaders()
                });
                if (await handleAuthError(response)) return;
                if (response.ok) {
                    currentObservationId = null;
                    document.getElementById('detail').innerHTML = '<div class="subtitle">Select a result to view details.</div>';
                    await loadStats();
                    await loadTagManager();
                    search();
                } else {
                    alert('Failed to delete observation');
                }
            }

            async function copyCitation() {
                if (!currentObservationId) return;
                const url = `${window.location.origin}/api/observation/${currentObservationId}`;
                try {
                    await navigator.clipboard.writeText(url);
                } catch (error) {
                    alert('Copy failed');
                }
            }

            async function deleteProject() {
                const project = document.getElementById('project').value;
                if (!project) {
                    alert('Select a project first.');
                    return;
                }
                if (!confirm(`Delete all observations for ${project}?`)) return;
                const response = await fetch('/api/projects/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                    body: JSON.stringify({ project })
                });
                if (await handleAuthError(response)) return;
                const result = await response.json();
                alert(`Deleted ${result.deleted} observations`);
                currentObservationId = null;
                document.getElementById('detail').innerHTML = '<div class="subtitle">Select a result to view details.</div>';
                await loadProjects();
                await loadTagManager();
                search();
            }

            async function exportData() {
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const params = new URLSearchParams();
                if (sessionId) {
                    params.append('session_id', sessionId);
                } else if (project) {
                    params.append('project', project);
                }
                const response = await fetch(`/api/export?${params.toString()}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                const data = await response.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                const suffix = sessionId ? safeSlug(sessionId, 'session') : safeSlug(project, 'all');
                link.href = url;
                link.download = `ai-mem-export-${suffix}.json`;
                document.body.appendChild(link);
                link.click();
                link.remove();
                URL.revokeObjectURL(url);
            }

            async function exportStats() {
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const type = document.getElementById('type').value;
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                const params = buildStatsParams({ day_limit: 60, tag_limit: 100, type_tag_limit: 3 });
                const response = await fetch(`/api/stats?${params.toString()}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                const data = await response.json();
                const rows = [['section', 'key', 'count']];
                rows.push(['total', 'total', data.total || 0]);
                rows.push(['recent_total', 'recent_total', data.recent_total || 0]);
                rows.push(['previous_total', 'previous_total', data.previous_total || 0]);
                rows.push(['trend_delta', 'trend_delta', data.trend_delta || 0]);
                if (data.trend_pct !== null && data.trend_pct !== undefined) {
                    rows.push(['trend_pct', 'trend_pct', data.trend_pct]);
                }
                (data.by_type || []).forEach(item => rows.push(['by_type', item.type, item.count]));
                (data.by_project || []).forEach(item => rows.push(['by_project', item.project, item.count]));
                (data.top_tags || []).forEach(item => rows.push(['top_tags', item.tag, item.count]));
                (data.top_tags_by_type || []).forEach(group => {
                    (group.tags || []).forEach(item => {
                        rows.push(['top_tags_by_type', `${group.type}:${item.tag}`, item.count]);
                    });
                });
                (data.by_day || []).forEach(item => rows.push(['by_day', item.day, item.count]));

                const csv = rows.map(row => row.map(csvEscape).join(',')).join('\n');
                const blob = new Blob([csv], { type: 'text/csv' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                const scopeSlug = sessionId
                    ? `session-${safeSlug(sessionId, 'session')}`
                    : safeSlug(project, 'all');
                const suffixParts = [
                    scopeSlug,
                    type ? safeSlug(type, 'type') : '',
                    dateStart ? safeSlug(dateStart, '') : '',
                    dateEnd ? safeSlug(dateEnd, '') : '',
                ].filter(Boolean);
                link.href = url;
                link.download = `ai-mem-stats-${suffixParts.join('-') || 'all'}.csv`;
                document.body.appendChild(link);
                link.click();
                link.remove();
                URL.revokeObjectURL(url);
            }

            async function exportStatsJson() {
                const project = document.getElementById('project').value;
                const sessionId = (document.getElementById('sessionId')?.value || '').trim();
                const type = document.getElementById('type').value;
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                const params = buildStatsParams({ day_limit: 60, tag_limit: 100, type_tag_limit: 3 });
                const response = await fetch(`/api/stats?${params.toString()}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                const data = await response.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                const scopeSlug = sessionId
                    ? `session-${safeSlug(sessionId, 'session')}`
                    : safeSlug(project, 'all');
                const suffixParts = [
                    scopeSlug,
                    type ? safeSlug(type, 'type') : '',
                    dateStart ? safeSlug(dateStart, '') : '',
                    dateEnd ? safeSlug(dateEnd, '') : '',
                ].filter(Boolean);
                link.href = url;
                link.download = `ai-mem-stats-${suffixParts.join('-') || 'all'}.json`;
                document.body.appendChild(link);
                link.click();
                link.remove();
                URL.revokeObjectURL(url);
            }

            function triggerImport() {
                document.getElementById('importFile').click();
            }

            async function importFile(event) {
                const file = event.target.files[0];
                if (!file) return;
                try {
                    const text = await file.text();
                    const data = JSON.parse(text);
                    let payload = null;
                    if (Array.isArray(data)) {
                        payload = { items: data };
                    } else if (data && Array.isArray(data.items)) {
                        payload = data;
                    }
                    if (!payload) {
                        alert('Invalid import file.');
                        return;
                    }
                    const project = document.getElementById('project').value;
                    if (project) {
                        payload.project = project;
                    }
                    const response = await fetch('/api/import', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                        body: JSON.stringify(payload)
                    });
                    if (await handleAuthError(response)) return;
                    const result = await response.json();
                    alert(`Imported ${result.imported} observations`);
                    await loadProjects();
                    await loadTagManager();
                    search();
                } catch (error) {
                    alert('Failed to import file.');
                } finally {
                    event.target.value = '';
                }
            }

            async function copyId() {
                if (!currentObservationId) return;
                try {
                    await navigator.clipboard.writeText(currentObservationId);
                } catch (error) {
                    alert('Copy failed');
                }
            }

            function saveToken() {
                const token = document.getElementById('token').value.trim();
                if (token) {
                    localStorage.setItem('ai-mem-token', token);
                    alert('Token saved.');
                } else {
                    localStorage.removeItem('ai-mem-token');
                    alert('Token cleared.');
                }
                restartStreamIfActive();
            }

            function loadToken() {
                const token = localStorage.getItem('ai-mem-token') || '';
                document.getElementById('token').value = token;
            }

            function bindProjectChange() {
                const select = document.getElementById('project');
                if (select.dataset.bound) return;
                select.addEventListener('change', async () => {
                    selectedProject = document.getElementById('project').value;
                    localStorage.setItem('ai-mem-selected-project', selectedProject);
                    loadAutoRefresh();
                    loadPulseToggle();
                    loadTimelineAnchor();
                    loadQuery();
                    await loadContextConfig();
                    loadLastMode();
                    await loadStats();
                    refreshSessions();
                    if (lastMode === 'timeline' && timelineQuery) {
                        const queryInput = document.getElementById('query');
                        if (queryInput) {
                            queryInput.value = timelineQuery;
                            persistQuery(timelineQuery);
                            updateQueryClearButton();
                        }
                    }
                    if (lastMode === 'timeline') {
                        await timeline();
                    } else {
                        await search();
                    }
                    restartStreamIfActive();
                });
                document.getElementById('type').addEventListener('change', () => {
                    persistFilters();
                    loadStats();
                    loadTagManager();
                    refreshResultsForFilters();
                    restartStreamIfActive();
                });
                document.getElementById('sessionId').addEventListener('change', () => {
                    persistFilters();
                    loadStats();
                    loadTagManager();
                    refreshResultsForFilters();
                    restartStreamIfActive();
                });
                document.getElementById('tagsFilter').addEventListener('change', () => {
                    persistFilters();
                    loadStats();
                    loadTagManager();
                    refreshResultsForFilters();
                    restartStreamIfActive();
                });
                document.getElementById('dateStart').addEventListener('change', () => {
                    persistFilters();
                    loadStats();
                    loadTagManager();
                    refreshResultsForFilters();
                });
                document.getElementById('dateEnd').addEventListener('change', () => {
                    persistFilters();
                    loadStats();
                    loadTagManager();
                    refreshResultsForFilters();
                });
                document.getElementById('autoRefresh').addEventListener('change', updateAutoRefresh);
                document.getElementById('refreshInterval').addEventListener('change', updateAutoRefresh);
                document.getElementById('pulseToggle').addEventListener('change', updatePulseToggle);
                document.getElementById('pulseGlobal').addEventListener('change', updatePulseGlobal);
                document.getElementById('queryGlobal').addEventListener('change', updateQueryGlobal);
                document.getElementById('autoGlobal').addEventListener('change', updateAutoGlobal);
                document.getElementById('contextTotal').addEventListener('change', updateContextConfig);
                document.getElementById('contextFull').addEventListener('change', updateContextConfig);
                document.getElementById('contextField').addEventListener('change', updateContextConfig);
                document.getElementById('contextTags').addEventListener('input', updateContextConfig);
                document.getElementById('contextTypes').addEventListener('input', updateContextConfig);
                document.getElementById('contextTokens').addEventListener('change', updateContextConfig);
                document.getElementById('contextWrap').addEventListener('change', updateContextConfig);
                document.getElementById('contextGlobal').addEventListener('change', updateContextConfig);
                document.getElementById('streamToggle').addEventListener('change', toggleStream);
                document.getElementById('streamQuery').addEventListener('input', () => {
                    const value = document.getElementById('streamQuery').value || '';
                    localStorage.setItem('ai-mem-stream-query', value);
                    restartStreamIfActive();
                });
                document.getElementById('sessionShowAll').addEventListener('change', () => {
                    const toggle = document.getElementById('sessionShowAll');
                    localStorage.setItem('ai-mem-sessions-show-all', String(toggle.checked));
                    refreshSessions();
                });
                document.getElementById('sessionGoalFilter').addEventListener('input', () => {
                    const value = document.getElementById('sessionGoalFilter').value || '';
                    localStorage.setItem('ai-mem-sessions-goal', value);
                    refreshSessions();
                });
                document.getElementById('sessionDateStart').addEventListener('change', () => {
                    const value = document.getElementById('sessionDateStart').value || '';
                    localStorage.setItem('ai-mem-sessions-date-start', value);
                    refreshSessions();
                });
                document.getElementById('sessionDateEnd').addEventListener('change', () => {
                    const value = document.getElementById('sessionDateEnd').value || '';
                    localStorage.setItem('ai-mem-sessions-date-end', value);
                    refreshSessions();
                });
                document.querySelectorAll('input[name="refreshMode"]').forEach(input => {
                    input.addEventListener('change', updateAutoRefresh);
                });
                document.getElementById('query').addEventListener('input', noteTyping);
                document.getElementById('query').addEventListener('keydown', noteTyping);
                document.getElementById('query').addEventListener('focus', noteTyping);
                document.getElementById('query').addEventListener('keydown', event => {
                    if (event.key === 'Escape') {
                        event.preventDefault();
                        clearQuery();
                    }
                });
                document.getElementById('query').addEventListener('input', () => {
                    timelineAnchorId = '';
                    timelineQuery = document.getElementById('query').value || '';
                    persistQuery(timelineQuery);
                    persistTimelineAnchor();
                    updateQueryClearButton();
                    updateAnchorPill();
                });
                document.getElementById('depthBefore').addEventListener('change', () => {
                    timelineDepthBefore = document.getElementById('depthBefore').value || timelineDepthBefore;
                    localStorage.setItem('ai-mem-timeline-depth-before', timelineDepthBefore);
                    updateResultsHeader();
                    if (lastMode === 'timeline') {
                        timeline();
                    }
                });
                document.getElementById('depthAfter').addEventListener('change', () => {
                    timelineDepthAfter = document.getElementById('depthAfter').value || timelineDepthAfter;
                    localStorage.setItem('ai-mem-timeline-depth-after', timelineDepthAfter);
                    updateResultsHeader();
                    if (lastMode === 'timeline') {
                        timeline();
                    }
                });
                document.getElementById('limit').addEventListener('change', () => {
                    listLimit = document.getElementById('limit').value || listLimit;
                    localStorage.setItem('ai-mem-list-limit', listLimit);
                    updateLimitWarning();
                });
                select.dataset.bound = 'true';
            }

            async function handleAuthError(response) {
                if (response.status === 401) {
                    alert('Unauthorized. Set API token in the sidebar.');
                    return true;
                }
                return false;
            }

            loadToken();
            loadStreamQuery();
            loadStreamToggle();
            loadSessionFilters();
            loadPulseToggle();
            loadAutoRefresh();
            loadTimelineAnchor();
            loadTimelineDepth();
            loadListLimit();
            loadSelectedProject();
            loadLastMode();
            loadSelectedFilters();
            loadSavedFilters();
            loadQuery();
            loadContextConfig().catch(err => console.warn('Context config load failed', err));
            loadStatsCollapse();
            bindProjectChange();
            loadProjects().then(async () => {
                if (lastMode === 'timeline') {
                    const queryInput = document.getElementById('query');
                    if (queryInput && timelineQuery) {
                        queryInput.value = timelineQuery;
                        persistQuery(timelineQuery);
                        updateQueryClearButton();
                    }
                    await timeline();
                    return;
                }
                await search();
            });
            window.addEventListener('resize', () => {
                updateSidebarLiveBadge();
                updateResultsHeader();
            });
        </script>
    </body>
    </html>
    """


@app.post("/api/memories")
def add_memory(mem: MemoryInput, request: Request):
    _check_token(request)
    try:
        obs = get_manager().add_observation(
            content=mem.content,
            obs_type=mem.obs_type,
            project=mem.project,
            session_id=mem.session_id,
            tags=mem.tags,
            metadata=mem.metadata,
            title=mem.title,
            summarize=mem.summarize,
        )
        if not obs:
            return {"status": "skipped", "reason": "private"}
        return obs.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/search")
def search_memories(
    request: Request,
    query: str,
    limit: int = 10,
    project: Optional[str] = None,
    obs_type: Optional[str] = None,
    session_id: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    tags: Optional[str] = None,
):
    _check_token(request)
    try:
        results = get_manager().search(
            query,
            limit=limit,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            tag_filters=_parse_list_param(tags),
        )
        return [item.model_dump() for item in results]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/timeline")
def get_timeline(
    request: Request,
    anchor_id: Optional[str] = None,
    query: Optional[str] = None,
    depth_before: int = 3,
    depth_after: int = 3,
    project: Optional[str] = None,
    obs_type: Optional[str] = None,
    session_id: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    tags: Optional[str] = None,
):
    _check_token(request)
    try:
        results = get_manager().timeline(
            anchor_id=anchor_id,
            query=query,
            depth_before=depth_before,
            depth_after=depth_after,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            tag_filters=_parse_list_param(tags),
        )
        return [item.model_dump() for item in results]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/context")
def context_alias(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    obs_type: Optional[str] = None,
    obs_types: Optional[str] = None,
    tags: Optional[str] = None,
    total: Optional[int] = None,
    full: Optional[int] = None,
    full_field: Optional[str] = None,
    show_tokens: Optional[bool] = None,
    wrap: Optional[bool] = None,
):
    return inject_context(
        request=request,
        project=project,
        session_id=session_id,
        query=query,
        obs_type=obs_type,
        obs_types=obs_types,
        tags=tags,
        total=total,
        full=full,
        full_field=full_field,
        show_tokens=show_tokens,
        wrap=wrap,
    )


@app.post("/api/context")
def context_alias_post(payload: ContextRequest, request: Request):
    return inject_context_post(payload, request)


@app.get("/api/context/config")
def context_config(request: Request):
    _check_token(request)
    return get_manager().config.context.model_dump()


@app.get("/api/context/preview")
def preview_context(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    obs_type: Optional[str] = None,
    obs_types: Optional[str] = None,
    tags: Optional[str] = None,
    total: Optional[int] = None,
    full: Optional[int] = None,
    full_field: Optional[str] = None,
    show_tokens: Optional[bool] = None,
    wrap: Optional[bool] = None,
):
    _check_token(request)
    context_text, _ = build_context(
        get_manager(),
        project=project,
        session_id=session_id,
        query=query,
        obs_type=obs_type,
        obs_types=_parse_list_param(obs_types),
        tag_filters=_parse_list_param(tags),
        total_count=total,
        full_count=full,
        full_field=full_field,
        show_tokens=show_tokens,
        wrap=False if wrap is None else wrap,
    )
    return Response(context_text, media_type="text/plain")


@app.post("/api/context/preview")
def preview_context_post(payload: ContextRequest, request: Request):
    _check_token(request)
    context_text, meta = build_context(
        get_manager(),
        project=payload.project,
        session_id=payload.session_id,
        query=payload.query,
        obs_type=payload.obs_type,
        obs_types=payload.obs_types,
        tag_filters=payload.tags,
        total_count=payload.total,
        full_count=payload.full,
        full_field=payload.full_field,
        show_tokens=payload.show_tokens,
        wrap=False if payload.wrap is None else payload.wrap,
    )
    return {"context": context_text, "metadata": meta}


@app.get("/api/context/inject")
def inject_context(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    obs_type: Optional[str] = None,
    obs_types: Optional[str] = None,
    tags: Optional[str] = None,
    total: Optional[int] = None,
    full: Optional[int] = None,
    full_field: Optional[str] = None,
    show_tokens: Optional[bool] = None,
    wrap: Optional[bool] = None,
):
    _check_token(request)
    context_text, _ = build_context(
        get_manager(),
        project=project,
        session_id=session_id,
        query=query,
        obs_type=obs_type,
        obs_types=_parse_list_param(obs_types),
        tag_filters=_parse_list_param(tags),
        total_count=total,
        full_count=full,
        full_field=full_field,
        show_tokens=show_tokens,
        wrap=wrap,
    )
    return Response(context_text, media_type="text/plain")


@app.post("/api/context/inject")
def inject_context_post(payload: ContextRequest, request: Request):
    _check_token(request)
    context_text, meta = build_context(
        get_manager(),
        project=payload.project,
        session_id=payload.session_id,
        query=payload.query,
        obs_type=payload.obs_type,
        obs_types=payload.obs_types,
        tag_filters=payload.tags,
        total_count=payload.total,
        full_count=payload.full,
        full_field=payload.full_field,
        show_tokens=payload.show_tokens,
        wrap=payload.wrap,
    )
    return {"context": context_text, "metadata": meta}


@app.post("/api/observations")
def get_observations(payload: ObservationIds, request: Request):
    _check_token(request)
    try:
        return get_manager().get_observations(payload.ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/observations/{obs_id}")
def get_observation(obs_id: str, request: Request):
    _check_token(request)
    try:
        results = get_manager().get_observations([obs_id])
        if not results:
            raise HTTPException(status_code=404, detail="Observation not found")
        return results[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.patch("/api/observations/{obs_id}")
def update_observation(obs_id: str, payload: ObservationUpdate, request: Request):
    _check_token(request)
    if payload.tags is None:
        raise HTTPException(status_code=400, detail="No updates provided")
    updated = get_manager().update_observation_tags(obs_id, payload.tags)
    if not updated:
        raise HTTPException(status_code=404, detail="Observation not found")
    return {"success": True}


@app.get("/api/observation/{obs_id}")
def get_observation_alias(obs_id: str, request: Request):
    return get_observation(obs_id, request)


@app.get("/api/projects")
def list_projects(request: Request):
    _check_token(request)
    return get_manager().list_projects()


@app.get("/api/sessions")
def list_sessions(
    request: Request,
    project: Optional[str] = None,
    active_only: bool = False,
    goal: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    limit: Optional[int] = None,
):
    _check_token(request)
    return get_manager().list_sessions(
        project=project,
        active_only=active_only,
        goal_query=goal,
        date_start=date_start,
        date_end=date_end,
        limit=limit,
    )


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, request: Request):
    _check_token(request)
    session = get_manager().get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/api/sessions/{session_id}/observations")
def get_session_observations(
    session_id: str,
    request: Request,
    limit: Optional[int] = None,
):
    _check_token(request)
    session = get_manager().get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return get_manager().export_observations(session_id=session_id, limit=limit)


@app.post("/api/sessions/start")
def start_session(payload: SessionStartRequest, request: Request):
    _check_token(request)
    project = payload.project or os.getcwd()
    session = get_manager().start_session(
        project=project,
        goal=payload.goal or "",
        session_id=payload.session_id,
    )
    return session.model_dump()


@app.post("/api/sessions/end")
def end_session(payload: SessionEndRequest, request: Request):
    _check_token(request)
    if payload.session_id:
        session = get_manager().end_session(payload.session_id)
    else:
        session = get_manager().end_latest_session(payload.project)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump()


@app.get("/api/stats")
def get_stats(
    request: Request,
    project: Optional[str] = None,
    obs_type: Optional[str] = None,
    session_id: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    tags: Optional[str] = None,
    tag_limit: Optional[int] = None,
    day_limit: Optional[int] = None,
    type_tag_limit: Optional[int] = None,
):
    _check_token(request)
    return get_manager().get_stats(
        project=project,
        obs_type=obs_type,
        session_id=session_id,
        date_start=date_start,
        date_end=date_end,
        tag_filters=_parse_list_param(tags),
        tag_limit=tag_limit if tag_limit is not None else 10,
        day_limit=day_limit if day_limit is not None else 14,
        type_tag_limit=type_tag_limit if type_tag_limit is not None else 3,
    )


@app.get("/api/tags")
def list_tags(
    request: Request,
    project: Optional[str] = None,
    obs_type: Optional[str] = None,
    session_id: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    tags: Optional[str] = None,
    limit: Optional[int] = 50,
):
    _check_token(request)
    return get_manager().list_tags(
        project=project,
        obs_type=obs_type,
        session_id=session_id,
        date_start=date_start,
        date_end=date_end,
        tag_filters=_parse_list_param(tags),
        limit=limit,
    )


@app.post("/api/tags/rename")
def rename_tag(payload: TagRenameRequest, request: Request):
    _check_token(request)
    old_tag = payload.old_tag.strip()
    new_tag = payload.new_tag.strip()
    if not old_tag or not new_tag:
        raise HTTPException(status_code=400, detail="Both old_tag and new_tag are required")
    updated = get_manager().rename_tag(
        old_tag=old_tag,
        new_tag=new_tag,
        project=payload.project,
        obs_type=payload.obs_type,
        session_id=payload.session_id,
        date_start=payload.date_start,
        date_end=payload.date_end,
        tag_filters=_parse_list_param(payload.tags),
    )
    return {"success": True, "updated": updated}


@app.post("/api/tags/add")
def add_tag(payload: TagAddRequest, request: Request):
    _check_token(request)
    value = payload.tag.strip()
    if not value:
        raise HTTPException(status_code=400, detail="tag is required")
    updated = get_manager().add_tag(
        tag=value,
        project=payload.project,
        obs_type=payload.obs_type,
        session_id=payload.session_id,
        date_start=payload.date_start,
        date_end=payload.date_end,
        tag_filters=_parse_list_param(payload.tags),
    )
    return {"success": True, "updated": updated}


@app.post("/api/tags/delete")
def delete_tag(payload: TagDeleteRequest, request: Request):
    _check_token(request)
    value = payload.tag.strip()
    if not value:
        raise HTTPException(status_code=400, detail="tag is required")
    updated = get_manager().delete_tag(
        tag=value,
        project=payload.project,
        obs_type=payload.obs_type,
        session_id=payload.session_id,
        date_start=payload.date_start,
        date_end=payload.date_end,
        tag_filters=_parse_list_param(payload.tags),
    )
    return {"success": True, "updated": updated}


@app.get("/api/observations")
def list_observations(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: Optional[int] = None,
):
    _check_token(request)
    return get_manager().export_observations(project=project, session_id=session_id, limit=limit)


@app.delete("/api/observations/{obs_id}")
def delete_observation(obs_id: str, request: Request):
    _check_token(request)
    if get_manager().delete_observation(obs_id):
        return {"success": True}
    raise HTTPException(status_code=404, detail="Observation not found")


@app.post("/api/projects/delete")
def delete_project(payload: ProjectDelete, request: Request):
    _check_token(request)
    deleted = get_manager().delete_project(payload.project)
    return {"success": True, "deleted": deleted}


@app.get("/api/export")
def export_observations(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: Optional[int] = None,
    format: Optional[str] = None,
):
    _check_token(request)
    data = get_manager().export_observations(project=project, session_id=session_id, limit=limit)
    fmt = (format or "json").lower()
    if fmt == "json":
        return data
    if fmt in {"jsonl", "ndjson"}:
        lines = "\n".join(json.dumps(item, ensure_ascii=True) for item in data)
        return Response(lines, media_type="application/x-ndjson")
    if fmt == "csv":
        output = io.StringIO()
        fields = [
            "id",
            "session_id",
            "project",
            "type",
            "title",
            "summary",
            "content",
            "created_at",
            "importance_score",
            "tags",
            "metadata",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item in data:
            writer.writerow(
                {
                    "id": item.get("id"),
                    "session_id": item.get("session_id"),
                    "project": item.get("project"),
                    "type": item.get("type"),
                    "title": item.get("title"),
                    "summary": item.get("summary"),
                    "content": item.get("content"),
                    "created_at": item.get("created_at"),
                    "importance_score": item.get("importance_score"),
                    "tags": json.dumps(item.get("tags") or [], ensure_ascii=True),
                    "metadata": json.dumps(item.get("metadata") or {}, ensure_ascii=True),
                }
            )
        return Response(output.getvalue(), media_type="text/csv")
    raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")


@app.post("/api/import")
def import_observations(payload: ImportPayload, request: Request):
    _check_token(request)
    manager = get_manager()
    imported = 0
    for item in payload.items:
        content = item.get("content")
        obs_type = item.get("type") or "note"
        if not content:
            continue
        item_project = item.get("project")
        session_id = item.get("session_id")
        if payload.project and item_project and payload.project != item_project:
            session_id = None
        result = manager.add_observation(
            content=content,
            obs_type=obs_type,
            project=payload.project or item_project,
            session_id=session_id,
            tags=item.get("tags") or [],
            metadata=item.get("metadata") or {},
            title=item.get("title"),
            summarize=False,
            dedupe=True,
            summary=item.get("summary"),
            created_at=item.get("created_at"),
            importance_score=item.get("importance_score", 0.5),
        )
        if result:
            imported += 1
    return {"success": True, "imported": imported}


@app.post("/api/summarize")
def summarize_project(payload: SummarizeRequest, request: Request):
    _check_token(request)
    result = get_manager().summarize_project(
        project=payload.project,
        session_id=payload.session_id,
        limit=payload.count,
        obs_type=payload.obs_type,
        store=payload.store,
        tags=payload.tags or None,
    )
    if not result:
        return {"status": "empty"}
    obs = result.get("observation")
    return {
        "status": "ok",
        "summary": result.get("summary", ""),
        "metadata": result.get("metadata"),
        "observation": obs.model_dump() if obs else None,
    }


@app.get("/api/health")
def health(request: Request):
    _check_token(request)
    return {"status": "ok"}


@app.get("/api/readiness")
def readiness(request: Request):
    _check_token(request)
    return {"status": "ok"}


@app.get("/api/version")
def version(request: Request):
    _check_token(request)
    return {"version": __version__}


@app.get("/api/stream")
async def stream_memories(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    obs_type: Optional[str] = None,
    tags: Optional[str] = None,
    token: Optional[str] = None,
):
    _check_token(request, query_token=token)
    manager = get_manager()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
    tag_filters = _parse_list_param(tags) or []
    query_text = (query or "").strip().lower()

    def _matches(obs: Dict[str, Any]) -> bool:
        if session_id and obs.get("session_id") != session_id:
            return False
        if project and obs.get("project") != project:
            return False
        if obs_type and obs.get("type") != obs_type:
            return False
        if tag_filters:
            obs_tags = obs.get("tags") or []
            if not any(tag in obs_tags for tag in tag_filters):
                return False
        if query_text:
            haystack_parts = [
                obs.get("summary") or "",
                obs.get("content") or "",
                obs.get("title") or "",
                " ".join(obs.get("tags") or []),
            ]
            haystack = " ".join(haystack_parts).lower()
            if query_text not in haystack:
                return False
        return True

    def _listener(obs_obj: Any) -> None:
        obs = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else {}
        if not obs or not _matches(obs):
            return
        summary_text = obs.get("summary") or obs.get("content") or ""
        token_estimate = estimate_tokens(summary_text)
        payload = {
            "id": obs.get("id"),
            "summary": summary_text,
            "project": obs.get("project") or "",
            "type": obs.get("type") or "",
            "created_at": obs.get("created_at"),
            "session_id": obs.get("session_id"),
            "tags": obs.get("tags") or [],
            "token_estimate": token_estimate,
        }
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    remove_listener = manager.add_listener(_listener)

    async def _event_stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(item, ensure_ascii=True)}\n\n"
        finally:
            remove_listener()

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
