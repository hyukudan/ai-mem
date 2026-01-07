import pytest
import tempfile
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ai_mem.db import DatabaseManager
from ai_mem.models import Observation, Session


@pytest.fixture
async def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
        await db.connect()
        yield db
        await db.close()


@pytest.mark.asyncio
async def test_fts_search(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Hello world from ai-mem",
        summary="Hello world",
    )
    await db.add_observation(obs)

    results = await db.search_observations_fts("Hello", project="proj", limit=5)
    assert results
    assert results[0].id == obs.id


@pytest.mark.asyncio
async def test_tag_filters(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs_alpha = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Alpha note",
        summary="Alpha note",
        tags=["alpha"],
        created_at=100.0,
    )
    obs_beta = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Beta note",
        summary="Beta note",
        tags=["beta"],
        created_at=200.0,
    )
    await db.add_observation(obs_alpha)
    await db.add_observation(obs_beta)

    results = await db.search_observations_fts(
        "note",
        project="proj",
        tag_filters=["alpha"],
        limit=5,
    )
    assert [item.id for item in results] == [obs_alpha.id]

    recent = await db.get_recent_observations(
        "proj",
        limit=5,
        tag_filters=["beta"],
    )
    assert [item.id for item in recent] == [obs_beta.id]

    stats = await db.get_stats(project="proj", tag_filters=["alpha"])
    assert stats["total"] == 1


@pytest.mark.asyncio
async def test_assets_are_stored_and_retrieved(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Asset note",
        summary="Asset note",
    )
    await db.add_observation(obs)
    await db.add_asset(
        observation_id=obs.id,
        asset_type="file",
        name="proof.txt",
        path="/tmp/proof.txt",
        content="diff content",
        metadata={"purpose": "test"},
    )
    assets = await db.get_assets_for_observation(obs.id)
    assert len(assets) == 1
    assert assets[0]["metadata"].get("purpose") == "test"
    stored = await db.get_observation(obs.id)
    assert len(stored.get("assets", [])) == 1


@pytest.mark.asyncio
async def test_list_observations_filters(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs_a = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Alpha",
        summary="Alpha",
        tags=["alpha"],
        created_at=100.0,
    )
    obs_b = Observation(
        session_id=session.id,
        project="proj",
        type="bugfix",
        content="Beta",
        summary="Beta",
        tags=["beta"],
        created_at=200.0,
    )
    obs_c = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Gamma",
        summary="Gamma",
        tags=["alpha", "beta"],
        created_at=300.0,
    )
    for obs in (obs_a, obs_b, obs_c):
        await db.add_observation(obs)

    filtered = await db.list_observations(project="proj", obs_type="note", tag_filters=["alpha"])
    assert [item.id for item in filtered] == [obs_c.id, obs_a.id]

    windowed = await db.list_observations(project="proj", date_start=150.0, date_end=250.0)
    assert [item.id for item in windowed] == [obs_b.id]


@pytest.mark.asyncio
async def test_update_observation_tags(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Tagged note",
        summary="Tagged note",
        tags=["alpha"],
    )
    await db.add_observation(obs)

    updated = await db.update_observation_tags(obs.id, ["alpha", "beta"])
    assert updated == 1
    stored = await db.get_observation(obs.id)
    assert stored["tags"] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_tag_counts_and_replace(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs_a = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Alpha",
        summary="Alpha",
        tags=["alpha", "beta"],
    )
    obs_b = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Beta",
        summary="Beta",
        tags=["beta"],
    )
    obs_c = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="No tags",
        summary="No tags",
        tags=[],
    )
    for obs in (obs_a, obs_b, obs_c):
        await db.add_observation(obs)

    counts_list = await db.get_tag_counts(project="proj", limit=None)
    counts = {item["tag"]: item["count"] for item in counts_list}
    assert counts.get("alpha") == 1
    assert counts.get("beta") == 2

    added = await db.add_tag("urgent", project="proj", tag_filters=["beta"])
    assert added == 2
    obs_a_added = await db.get_observation(obs_a.id)
    obs_b_added = await db.get_observation(obs_b.id)
    assert "urgent" in obs_a_added["tags"]
    assert "urgent" in obs_b_added["tags"]

    updated = await db.replace_tag("beta", "gamma", project="proj", tag_filters=["alpha"])
    assert updated == 1
    obs_a_updated = await db.get_observation(obs_a.id)
    obs_b_updated = await db.get_observation(obs_b.id)
    assert "gamma" in obs_a_updated["tags"]
    assert "beta" not in obs_a_updated["tags"]
    assert "beta" in obs_b_updated["tags"]
    assert "urgent" in obs_b_updated["tags"]

    updated_all = await db.replace_tag("beta", "gamma", project="proj")
    assert updated_all == 1
    obs_b_updated = await db.get_observation(obs_b.id)
    assert "gamma" in obs_b_updated["tags"]
    assert "urgent" in obs_b_updated["tags"]

    removed = await db.replace_tag("alpha", None, project="proj")
    assert removed == 1
    obs_a_final = await db.get_observation(obs_a.id)
    assert "alpha" not in obs_a_final["tags"]


@pytest.mark.asyncio
async def test_stats(db):
    session_a = Session(project="proj-a")
    session_b = Session(project="proj-b")
    await db.add_session(session_a)
    await db.add_session(session_b)

    obs_a1 = Observation(
        session_id=session_a.id,
        project="proj-a",
        type="note",
        content="Alpha note",
        summary="Alpha",
        tags=["alpha", "shared"],
    )
    obs_a2 = Observation(
        session_id=session_a.id,
        project="proj-a",
        type="bugfix",
        content="Fix bug",
        summary="Fix",
        tags=["shared"],
    )
    obs_b1 = Observation(
        session_id=session_b.id,
        project="proj-b",
        type="note",
        content="Beta note",
        summary="Beta",
        tags=["beta"],
    )
    for obs in (obs_a1, obs_a2, obs_b1):
        await db.add_observation(obs)

    stats_all = await db.get_stats()
    assert stats_all["total"] == 3
    assert stats_all["by_project"][0]["project"] == "proj-a"
    assert stats_all["by_project"][0]["count"] == 2

    stats_a = await db.get_stats(project="proj-a")
    assert stats_a["total"] == 2
    assert stats_a["by_project"][0]["project"] == "proj-a"
    assert stats_a["by_project"][0]["count"] == 2
    assert stats_a["top_tags"][0]["tag"] == "shared"
    assert stats_a["top_tags"][0]["count"] == 2


@pytest.mark.asyncio
async def test_stats_date_filter(db):
    session = Session(project="proj")
    await db.add_session(session)

    older = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Old",
        summary="Old",
        created_at=100.0,
    )
    newer = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="New",
        summary="New",
        created_at=200.0,
    )
    await db.add_observation(older)
    await db.add_observation(newer)

    stats = await db.get_stats(date_start=150.0)
    assert stats["total"] == 1
    assert stats["by_project"][0]["project"] == "proj"
    assert stats["by_project"][0]["count"] == 1
    assert stats["by_day"][0]["count"] == 1
    assert stats["recent_total"] == 1


@pytest.mark.asyncio
async def test_stats_by_day(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs_early = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Day one",
        summary="Day one",
        created_at=86400.0,
    )
    obs_late = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Day two",
        summary="Day two",
        created_at=172800.0,
    )
    await db.add_observation(obs_early)
    await db.add_observation(obs_late)

    stats = await db.get_stats(day_limit=5)
    day_counts = {item["day"]: item["count"] for item in stats["by_day"]}
    assert day_counts.get("1970-01-02") == 1
    assert day_counts.get("1970-01-03") == 1
    assert stats["recent_total"] == 2


@pytest.mark.asyncio
async def test_stats_obs_type(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs_note = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Note",
        summary="Note",
        tags=["alpha"],
    )
    obs_bug = Observation(
        session_id=session.id,
        project="proj",
        type="bugfix",
        content="Bugfix",
        summary="Bugfix",
        tags=["beta"],
    )
    await db.add_observation(obs_note)
    await db.add_observation(obs_bug)

    stats = await db.get_stats(obs_type="bugfix")
    assert stats["total"] == 1
    assert stats["by_type"][0]["type"] == "bugfix"
    assert stats["top_tags"][0]["tag"] == "beta"


@pytest.mark.asyncio
async def test_stats_top_tags_by_type(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs_note = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Note",
        summary="Note",
        tags=["alpha", "shared"],
    )
    obs_bug = Observation(
        session_id=session.id,
        project="proj",
        type="bugfix",
        content="Bugfix",
        summary="Bugfix",
        tags=["beta", "shared"],
    )
    await db.add_observation(obs_note)
    await db.add_observation(obs_bug)

    stats = await db.get_stats(type_tag_limit=1)
    grouped = {item["type"]: item["tags"][0]["tag"] for item in stats["top_tags_by_type"]}
    assert grouped.get("note") == "alpha"
    assert grouped.get("bugfix") == "beta"


@pytest.mark.asyncio
async def test_before_after_filters(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs_early = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Early",
        summary="Early",
        created_at=100.0,
    )
    obs_mid = Observation(
        session_id=session.id,
        project="proj",
        type="bugfix",
        content="Mid",
        summary="Mid",
        created_at=200.0,
    )
    obs_late = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Late",
        summary="Late",
        created_at=300.0,
    )
    for obs in (obs_early, obs_mid, obs_late):
        await db.add_observation(obs)

    before_notes = await db.get_observations_before(
        project="proj",
        anchor_time=250.0,
        limit=10,
        obs_type="note",
    )
    assert [item.id for item in before_notes] == [obs_early.id]

    after_notes = await db.get_observations_after(
        project="proj",
        anchor_time=250.0,
        limit=10,
        obs_type="note",
    )
    assert [item.id for item in after_notes] == [obs_late.id]

    before_date = await db.get_observations_before(
        project="proj",
        anchor_time=350.0,
        limit=10,
        date_start=250.0,
    )
    assert [item.id for item in before_date] == [obs_late.id]

    after_date = await db.get_observations_after(
        project="proj",
        anchor_time=150.0,
        limit=10,
        date_end=250.0,
    )
    assert [item.id for item in after_date] == [obs_mid.id]


@pytest.mark.asyncio
async def test_stats_trend(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs_prev = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Prev",
        summary="Prev",
        created_at=50000.0,
    )
    obs_recent_a = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Recent A",
        summary="Recent A",
        created_at=150000.0,
    )
    obs_recent_b = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Recent B",
        summary="Recent B",
        created_at=160000.0,
    )
    await db.add_observation(obs_prev)
    await db.add_observation(obs_recent_a)
    await db.add_observation(obs_recent_b)

    stats = await db.get_stats(day_limit=1)
    assert stats["recent_total"] == 2
    assert stats["previous_total"] == 1
    assert stats["trend_delta"] == 1


@pytest.mark.asyncio
async def test_sessions_list(db):
    active = Session(project="proj-a", goal="Alpha goal", start_time=100.0)
    ended = Session(project="proj-a", goal="Beta goal", start_time=200.0, end_time=123.0)
    await db.add_session(active)
    await db.add_session(ended)

    sessions_all = await db.list_sessions(project="proj-a")
    assert len(sessions_all) == 2

    sessions_active = await db.list_sessions(project="proj-a", active_only=True)
    assert len(sessions_active) == 1
    assert sessions_active[0]["id"] == active.id

    goal_match = await db.list_sessions(project="proj-a", goal_query="alpha")
    assert len(goal_match) == 1
    assert goal_match[0]["id"] == active.id

    recent = await db.list_sessions(project="proj-a", date_start=150.0)
    assert len(recent) == 1
    assert recent[0]["id"] == ended.id


@pytest.mark.asyncio
async def test_observations_by_session(db):
    session_a = Session(project="proj-a")
    session_b = Session(project="proj-a")
    await db.add_session(session_a)
    await db.add_session(session_b)

    obs_a = Observation(
        session_id=session_a.id,
        project="proj-a",
        type="note",
        content="A",
        summary="A",
    )
    obs_b = Observation(
        session_id=session_b.id,
        project="proj-a",
        type="note",
        content="B",
        summary="B",
    )
    await db.add_observation(obs_a)
    await db.add_observation(obs_b)

    results = await db.list_observations(session_id=session_a.id)
    assert len(results) == 1
    assert results[0].id == obs_a.id


# ============================================
# Event Idempotency Tests
# ============================================

@pytest.mark.asyncio
async def test_event_idempotency_not_processed(db):
    """Test that an unprocessed event_id returns None."""
    result = await db.check_event_processed("non-existent-event-id")
    assert result is None


@pytest.mark.asyncio
async def test_event_idempotency_record_and_check(db):
    """Test recording and checking a processed event."""
    event_id = "test-event-123"
    observation_id = "obs-456"
    host = "claude-code"

    # Record the event as processed
    await db.record_event_processed(event_id, observation_id, host)

    # Check should return the observation ID
    result = await db.check_event_processed(event_id)
    assert result == observation_id


@pytest.mark.asyncio
async def test_event_idempotency_duplicate_record(db):
    """Test that recording the same event_id twice doesn't raise an error."""
    event_id = "duplicate-event"

    # Record twice - should not raise (INSERT OR IGNORE)
    await db.record_event_processed(event_id, "obs-1", "host-1")
    await db.record_event_processed(event_id, "obs-2", "host-2")

    # Should return the first observation_id (first insert wins)
    result = await db.check_event_processed(event_id)
    assert result == "obs-1"


@pytest.mark.asyncio
async def test_event_idempotency_multiple_events(db):
    """Test multiple different event_ids."""
    events = [
        ("event-a", "obs-a", "claude-code"),
        ("event-b", "obs-b", "gemini"),
        ("event-c", "obs-c", "cursor"),
    ]

    for event_id, obs_id, host in events:
        await db.record_event_processed(event_id, obs_id, host)

    # Each should return its own observation_id
    assert await db.check_event_processed("event-a") == "obs-a"
    assert await db.check_event_processed("event-b") == "obs-b"
    assert await db.check_event_processed("event-c") == "obs-c"


@pytest.mark.asyncio
async def test_event_idempotency_cleanup_old(db):
    """Test cleanup of old event IDs."""
    import time

    # Record some events
    await db.record_event_processed("recent-event", "obs-recent", "host")

    # Manually insert an old event (simulate 60 days ago)
    old_timestamp = time.time() - (60 * 24 * 60 * 60)
    await db.conn.execute(
        "INSERT INTO event_idempotency (event_id, observation_id, host, processed_at) VALUES (?, ?, ?, ?)",
        ("old-event", "obs-old", "host", old_timestamp),
    )
    await db.conn.commit()

    # Both should exist initially
    assert await db.check_event_processed("recent-event") is not None
    assert await db.check_event_processed("old-event") is not None

    # Cleanup with 30 days threshold
    deleted = await db.cleanup_old_event_ids(max_age_days=30)
    assert deleted == 1  # Only the old event should be deleted

    # Recent should still exist, old should be gone
    assert await db.check_event_processed("recent-event") is not None
    assert await db.check_event_processed("old-event") is None


@pytest.mark.asyncio
async def test_event_idempotency_cleanup_nothing_to_delete(db):
    """Test cleanup when there's nothing to delete."""
    await db.record_event_processed("event-1", "obs-1", "host")

    # Cleanup with 30 days - nothing should be deleted (event is fresh)
    deleted = await db.cleanup_old_event_ids(max_age_days=30)
    assert deleted == 0

    # Event should still exist
    assert await db.check_event_processed("event-1") == "obs-1"
