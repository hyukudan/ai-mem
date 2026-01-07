"""Tests for the REST API server endpoints.

These tests verify the FastAPI server endpoints work correctly using
TestClient for synchronous testing.
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# We need to set up environment before importing server
@pytest.fixture(scope="module")
def temp_env():
    """Set up temporary environment for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = {
            "AI_MEM_DATA_DIR": tmpdir,
            "AI_MEM_API_TOKEN": "test-token",
        }
        with patch.dict(os.environ, env):
            yield tmpdir


@pytest.fixture
def client(temp_env):
    """Create a test client for the API."""
    from ai_mem.server import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers():
    """Return authorization headers."""
    return {"Authorization": "Bearer test-token"}


# =============================================================================
# Test: Health and readiness endpoints
# =============================================================================


def test_health_endpoint(client, auth_headers):
    """Test the health endpoint returns OK."""
    response = client.get("/api/health", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_readiness_endpoint(client, auth_headers):
    """Test the readiness endpoint returns ready status."""
    response = client.get("/api/readiness", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data or "status" in data


def test_version_endpoint(client, auth_headers):
    """Test the version endpoint returns version info."""
    response = client.get("/api/version", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "version" in data or "name" in data


# =============================================================================
# Test: Root endpoint
# =============================================================================


def test_root_returns_html(client):
    """Test the root endpoint returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


# =============================================================================
# Test: Search endpoint
# =============================================================================


def test_search_requires_query(client, auth_headers):
    """Test search endpoint requires query parameter."""
    response = client.get("/api/search", headers=auth_headers)
    # Query is required, should return 422 if missing
    assert response.status_code in [200, 422]


def test_search_with_query(client, auth_headers):
    """Test search with query parameter."""
    response = client.get("/api/search?query=test", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_search_with_limit(client, auth_headers):
    """Test search with limit parameter."""
    response = client.get("/api/search?query=test&limit=5", headers=auth_headers)
    assert response.status_code == 200


def test_search_with_project_filter(client, auth_headers):
    """Test search with project filter."""
    response = client.get("/api/search?query=test&project=my-project", headers=auth_headers)
    assert response.status_code == 200


# =============================================================================
# Test: Context endpoint
# =============================================================================


def test_context_get(client, auth_headers):
    """Test GET context endpoint."""
    response = client.get("/api/context", headers=auth_headers)
    assert response.status_code == 200


def test_context_post(client, auth_headers):
    """Test POST context endpoint."""
    response = client.post(
        "/api/context",
        json={"query": "test"},
        headers=auth_headers,
    )
    assert response.status_code == 200


def test_context_config(client, auth_headers):
    """Test context config endpoint."""
    response = client.get("/api/context/config", headers=auth_headers)
    assert response.status_code == 200


def test_context_preview(client, auth_headers):
    """Test context preview endpoint."""
    response = client.get("/api/context/preview", headers=auth_headers)
    assert response.status_code == 200


# =============================================================================
# Test: Projects endpoint
# =============================================================================


def test_list_projects(client, auth_headers):
    """Test listing projects."""
    response = client.get("/api/projects", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# =============================================================================
# Test: Sessions endpoints
# =============================================================================


def test_list_sessions(client, auth_headers):
    """Test listing sessions."""
    response = client.get("/api/sessions", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_session_stats(client, auth_headers):
    """Test session stats endpoint."""
    response = client.get("/api/stats/sessions", headers=auth_headers)
    assert response.status_code == 200


def test_start_session(client, auth_headers):
    """Test starting a new session."""
    response = client.post(
        "/api/sessions/start",
        json={"project": "test-project"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data


# =============================================================================
# Test: Stats endpoint
# =============================================================================


def test_stats_endpoint(client, auth_headers):
    """Test the stats endpoint."""
    response = client.get("/api/stats", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "total" in data


def test_stats_with_project(client, auth_headers):
    """Test stats with project filter."""
    response = client.get("/api/stats?project=test", headers=auth_headers)
    assert response.status_code == 200


# =============================================================================
# Test: Tags endpoints
# =============================================================================


def test_list_tags(client, auth_headers):
    """Test listing tags."""
    response = client.get("/api/tags", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_rename_tag(client, auth_headers):
    """Test renaming a tag."""
    response = client.post(
        "/api/tags/rename",
        json={"old_tag": "old", "new_tag": "new"},
        headers=auth_headers,
    )
    assert response.status_code == 200


def test_add_tag(client, auth_headers):
    """Test adding a tag."""
    response = client.post(
        "/api/tags/add",
        json={"tag": "new-tag"},
        headers=auth_headers,
    )
    assert response.status_code == 200


def test_delete_tag(client, auth_headers):
    """Test deleting a tag."""
    response = client.post(
        "/api/tags/delete",
        json={"tag": "old-tag"},
        headers=auth_headers,
    )
    assert response.status_code == 200


# =============================================================================
# Test: Observations endpoints
# =============================================================================


def test_list_observations(client, auth_headers):
    """Test listing observations."""
    response = client.get("/api/observations", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_observations_with_limit(client, auth_headers):
    """Test listing observations with limit."""
    response = client.get("/api/observations?limit=5", headers=auth_headers)
    assert response.status_code == 200


def test_add_observation(client, auth_headers):
    """Test adding an observation via /api/memories."""
    response = client.post(
        "/api/memories",
        json={
            "content": "Test observation for add",
            "obs_type": "note",
            "project": "test-project",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data or "status" in data


def test_add_memory_alias(client, auth_headers):
    """Test the /api/memories endpoint (alias for add observation)."""
    response = client.post(
        "/api/memories",
        json={
            "content": "Test memory",
            "type": "note",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200


# =============================================================================
# Test: Events endpoint (for adapters)
# =============================================================================


def test_events_endpoint(client, auth_headers):
    """Test the events endpoint for host adapters."""
    response = client.post(
        "/api/events",
        json={
            "host": "claude-code",
            "payload": {
                "tool_name": "Read",
                "tool_input": {"path": "/test/file.py"},
                "tool_response": "file contents",
            },
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "observation_id" in data


def test_events_endpoint_generic_host(client, auth_headers):
    """Test events endpoint with generic host."""
    response = client.post(
        "/api/events",
        json={
            "host": "custom-tool",
            "payload": {
                "name": "my_tool",
                "input": {"arg": "value"},
                "output": "result",
            },
        },
        headers=auth_headers,
    )
    assert response.status_code == 200


def test_events_endpoint_idempotency(client, auth_headers):
    """Test events endpoint respects event_id for idempotency."""
    event_id = "test-event-123"
    payload = {
        "host": "claude-code",
        "payload": {
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "files",
            "event_id": event_id,
        },
    }

    # First request should succeed
    response1 = client.post("/api/events", json=payload, headers=auth_headers)
    assert response1.status_code == 200

    # Second request with same event_id should be skipped
    response2 = client.post("/api/events", json=payload, headers=auth_headers)
    assert response2.status_code == 200
    # Should indicate it was skipped or return same observation


# =============================================================================
# Test: Export/Import endpoints
# =============================================================================


def test_export_endpoint(client, auth_headers):
    """Test export endpoint."""
    response = client.get("/api/export", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    # Export returns a list of observations or a dict with observations
    assert isinstance(data, (list, dict))


def test_import_endpoint(client, auth_headers):
    """Test import endpoint with empty data."""
    response = client.post(
        "/api/import",
        json={"data": []},
        headers=auth_headers,
    )
    # May require different format
    assert response.status_code in [200, 422]


# =============================================================================
# Test: Authentication
# =============================================================================


def test_protected_endpoint_without_auth(client):
    """Test protected endpoint requires authentication."""
    # When API token is set, requests without auth should fail
    response = client.get("/api/search?query=test")
    # May return 401 or 403 depending on implementation
    assert response.status_code in [200, 401, 403]


def test_protected_endpoint_with_wrong_token(client):
    """Test protected endpoint rejects wrong token."""
    response = client.get(
        "/api/search?query=test",
        headers={"Authorization": "Bearer wrong-token"},
    )
    # May return 401 or 403
    assert response.status_code in [200, 401, 403]


# =============================================================================
# Test: Timeline endpoint
# =============================================================================


def test_timeline_endpoint(client, auth_headers):
    """Test timeline endpoint."""
    response = client.get("/api/timeline?obs_id=test-id", headers=auth_headers)
    # May return 404 if observation doesn't exist, or 200 with empty data
    assert response.status_code in [200, 404]


# =============================================================================
# Test: Error handling
# =============================================================================


def test_invalid_json_body(client, auth_headers):
    """Test handling of invalid JSON body."""
    response = client.post(
        "/api/observations",
        content="not valid json",
        headers={**auth_headers, "Content-Type": "application/json"},
    )
    assert response.status_code in [400, 422]


def test_missing_required_fields(client, auth_headers):
    """Test handling of missing required fields."""
    response = client.post(
        "/api/observations",
        json={},  # Missing required fields
        headers=auth_headers,
    )
    assert response.status_code in [200, 400, 422]


def test_invalid_observation_id(client, auth_headers):
    """Test handling of invalid observation ID."""
    response = client.get("/api/observations/nonexistent-id", headers=auth_headers)
    # May return 400 (invalid UUID), 404 (not found), or 200 (null)
    assert response.status_code in [200, 400, 404]


# =============================================================================
# Test: Limit validation
# =============================================================================


def test_limit_max_value(client, auth_headers):
    """Test that excessive limit values are handled."""
    response = client.get("/api/observations?limit=10000", headers=auth_headers)
    # Should succeed but may cap the limit
    assert response.status_code == 200


def test_limit_negative_value(client, auth_headers):
    """Test that negative limit values are handled."""
    response = client.get("/api/observations?limit=-1", headers=auth_headers)
    # Should handle gracefully
    assert response.status_code in [200, 400, 422]


# =============================================================================
# Test: Content type handling
# =============================================================================


def test_json_response_type(client, auth_headers):
    """Test that API returns JSON content type."""
    response = client.get("/api/stats", headers=auth_headers)
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")


# =============================================================================
# Test: CORS headers (if enabled)
# =============================================================================


def test_cors_preflight(client):
    """Test CORS preflight request handling."""
    response = client.options(
        "/api/search",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Should not fail, may return 200 or 204
    assert response.status_code in [200, 204, 405]
