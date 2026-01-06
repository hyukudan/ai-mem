import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

import uvicorn

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


def _check_token(request: Request) -> None:
    if not _api_token:
        return
    auth = request.headers.get("authorization") or ""
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    if not token:
        token = request.headers.get("x-ai-mem-token", "")
    if token != _api_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


class MemoryInput(BaseModel):
    content: str
    obs_type: str = "note"
    project: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    title: Optional[str] = None
    summarize: bool = True


class ObservationIds(BaseModel):
    ids: List[str]


class ProjectDelete(BaseModel):
    project: str


class ImportPayload(BaseModel):
    items: List[Dict[str, Any]]
    project: Optional[str] = None


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
            .pill {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: #f5e8da;
                color: #7a4a2f;
                font-size: 12px;
                margin-right: 6px;
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
                <div class="subtitle">API: /api/search, /api/timeline, /api/observations</div>
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
                        </select>
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
            let selectedType = '';
            let selectedDateStart = '';
            let selectedDateEnd = '';

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
                loadAutoRefresh();
                loadPulseToggle();
                loadLastMode();
                loadTimelineAnchor();
                loadQuery();
            }

            function buildQueryParams() {
                const query = document.getElementById('query').value || " ";
                const project = document.getElementById('project').value;
                const type = document.getElementById('type').value;
                const limit = document.getElementById('limit').value;
                listLimit = limit || listLimit;
                localStorage.setItem('ai-mem-list-limit', listLimit);
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                const params = new URLSearchParams({ query });
                if (project) params.append('project', project);
                if (type) params.append('obs_type', type);
                if (limit) params.append('limit', limit);
                if (dateStart) params.append('date_start', dateStart);
                if (dateEnd) params.append('date_end', dateEnd);
                return params.toString();
            }

            function buildStatsParams(options = {}) {
                const project = document.getElementById('project').value;
                const type = document.getElementById('type').value;
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                const params = new URLSearchParams();
                if (project) params.append('project', project);
                if (type) params.append('obs_type', type);
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
                const autoBadge = live
                    ? `<span class="auto-mode-badge" title="Auto-refresh: ${fullModeLabel} • ${interval}s • ${autoScope}">${modeLabel}</span>`
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
                const type = document.getElementById('type').value;
                const dateStart = document.getElementById('dateStart').value;
                const dateEnd = document.getElementById('dateEnd').value;
                if (project) dots.push({ label: 'project', value: project });
                if (type) dots.push({ label: 'type', value: type });
                if (dateStart || dateEnd) {
                    dots.push({ label: 'date', value: `${dateStart || '…'} → ${dateEnd || '…'}` });
                }
                if (query) dots.push({ label: 'query', value: query });
                const iconMap = {
                    project: 'P',
                    type: 'T',
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
                document.getElementById('type').value = '';
                document.getElementById('dateStart').value = '';
                document.getElementById('dateEnd').value = '';
                timelineAnchorId = '';
                timelineQuery = '';
                selectedProject = '';
                selectedType = '';
                selectedDateStart = '';
                selectedDateEnd = '';
                listLimit = '10';
                timelineDepthBefore = '3';
                timelineDepthAfter = '3';
                document.getElementById('limit').value = listLimit;
                document.getElementById('depthBefore').value = timelineDepthBefore;
                document.getElementById('depthAfter').value = timelineDepthAfter;
                localStorage.removeItem('ai-mem-selected-project');
                localStorage.removeItem('ai-mem-selected-type');
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
                search();
            }

            function persistFilters() {
                selectedType = document.getElementById('type').value || '';
                selectedDateStart = document.getElementById('dateStart').value || '';
                selectedDateEnd = document.getElementById('dateEnd').value || '';
                localStorage.setItem('ai-mem-selected-type', selectedType);
                localStorage.setItem('ai-mem-date-start', selectedDateStart);
                localStorage.setItem('ai-mem-date-end', selectedDateEnd);
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
                selectedType = localStorage.getItem('ai-mem-selected-type') || '';
                selectedDateStart = localStorage.getItem('ai-mem-date-start') || '';
                selectedDateEnd = localStorage.getItem('ai-mem-date-end') || '';
                if (selectedType) {
                    document.getElementById('type').value = selectedType;
                }
                document.getElementById('dateStart').value = selectedDateStart;
                document.getElementById('dateEnd').value = selectedDateEnd;
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
                    div.innerHTML = `
                        <div>${mem.summary || '(no summary)'}</div>
                        <div class="meta">
                            ${mem.project || 'Global'} • ${mem.type || 'note'} • ${mem.id}
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
                if (project) params.append('project', project);
                if (type) params.append('obs_type', type);
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
                const tags = (mem.tags || []).map(tag => `<span class="pill">${tag}</span>`).join('');
                detail.innerHTML = `
                    <div><strong>${mem.summary || '(no summary)'}</strong></div>
                    <div class="meta">${mem.project || 'Global'} • ${mem.type || 'note'} • ${mem.id}</div>
                    <div>${tags || '<span class="subtitle">No tags</span>'}</div>
                    <div class="button-row">
                        <button onclick="copyId()">Copy ID</button>
                        <button class="secondary" onclick="deleteObservation()">Delete</button>
                    </div>
                    <div class="accent">Content</div>
                    <pre>${mem.content || ''}</pre>
                `;
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
                    search();
                } else {
                    alert('Failed to delete observation');
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
                search();
            }

            async function exportData() {
                const project = document.getElementById('project').value;
                const params = new URLSearchParams();
                if (project) params.append('project', project);
                const response = await fetch(`/api/export?${params.toString()}`, { headers: getAuthHeaders() });
                if (await handleAuthError(response)) return;
                const data = await response.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                const suffix = safeSlug(project, 'all');
                link.href = url;
                link.download = `ai-mem-export-${suffix}.json`;
                document.body.appendChild(link);
                link.click();
                link.remove();
                URL.revokeObjectURL(url);
            }

            async function exportStats() {
                const project = document.getElementById('project').value;
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
                const suffixParts = [
                    safeSlug(project, 'all'),
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
                const suffixParts = [
                    safeSlug(project, 'all'),
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
                    loadLastMode();
                    await loadStats();
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
                });
                document.getElementById('type').addEventListener('change', () => {
                    persistFilters();
                    loadStats();
                    refreshResultsForFilters();
                });
                document.getElementById('dateStart').addEventListener('change', () => {
                    persistFilters();
                    loadStats();
                    refreshResultsForFilters();
                });
                document.getElementById('dateEnd').addEventListener('change', () => {
                    persistFilters();
                    loadStats();
                    refreshResultsForFilters();
                });
                document.getElementById('autoRefresh').addEventListener('change', updateAutoRefresh);
                document.getElementById('refreshInterval').addEventListener('change', updateAutoRefresh);
                document.getElementById('pulseToggle').addEventListener('change', updatePulseToggle);
                document.getElementById('pulseGlobal').addEventListener('change', updatePulseGlobal);
                document.getElementById('queryGlobal').addEventListener('change', updateQueryGlobal);
                document.getElementById('autoGlobal').addEventListener('change', updateAutoGlobal);
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
            loadPulseToggle();
            loadAutoRefresh();
            loadTimelineAnchor();
            loadTimelineDepth();
            loadListLimit();
            loadSelectedProject();
            loadLastMode();
            loadSelectedFilters();
            loadQuery();
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
            tags=mem.tags,
            metadata=mem.metadata,
            title=mem.title,
            summarize=mem.summarize,
        )
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
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
):
    _check_token(request)
    try:
        results = get_manager().search(
            query,
            limit=limit,
            project=project,
            obs_type=obs_type,
            date_start=date_start,
            date_end=date_end,
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
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
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
            date_start=date_start,
            date_end=date_end,
        )
        return [item.model_dump() for item in results]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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


@app.get("/api/projects")
def list_projects(request: Request):
    _check_token(request)
    return get_manager().list_projects()


@app.get("/api/stats")
def get_stats(
    request: Request,
    project: Optional[str] = None,
    obs_type: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    tag_limit: Optional[int] = None,
    day_limit: Optional[int] = None,
    type_tag_limit: Optional[int] = None,
):
    _check_token(request)
    return get_manager().get_stats(
        project=project,
        obs_type=obs_type,
        date_start=date_start,
        date_end=date_end,
        tag_limit=tag_limit if tag_limit is not None else 10,
        day_limit=day_limit if day_limit is not None else 14,
        type_tag_limit=type_tag_limit if type_tag_limit is not None else 3,
    )


@app.get("/api/observations")
def list_observations(
    request: Request,
    project: Optional[str] = None,
    limit: Optional[int] = None,
):
    _check_token(request)
    return get_manager().export_observations(project=project, limit=limit)


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
    limit: Optional[int] = None,
):
    _check_token(request)
    return get_manager().export_observations(project=project, limit=limit)


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
        manager.add_observation(
            content=content,
            obs_type=obs_type,
            project=payload.project or item.get("project"),
            tags=item.get("tags") or [],
            metadata=item.get("metadata") or {},
            title=item.get("title"),
            summarize=False,
            dedupe=True,
            summary=item.get("summary"),
            created_at=item.get("created_at"),
            importance_score=item.get("importance_score", 0.5),
        )
        imported += 1
    return {"success": True, "imported": imported}


@app.get("/api/health")
def health(request: Request):
    _check_token(request)
    return {"status": "ok"}


def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
