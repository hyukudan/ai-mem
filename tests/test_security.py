import pytest
import tempfile
import sys
import os
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ai_mem.db import DatabaseManager
from ai_mem.privacy import strip_memory_tags, _count_tags
from ai_mem.models import Observation, Session

@pytest.fixture
async def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(f"{tmpdir}/ai-mem-security.sqlite")
        await db.connect()
        yield db
        await db.close()

@pytest.mark.asyncio
async def test_fts_injection_sanitization(db):
    session = Session(project="proj")
    await db.add_session(session)

    obs = Observation(
        session_id=session.id,
        project="proj",
        type="note",
        content="Testing security AND operators in FTS",
        summary="Security tests",
    )
    await db.add_observation(obs)

    # This should be treated as a literal phrase, not an AND operator
    results = await db.search_observations_fts("security AND operators", project="proj")
    assert len(results) == 1
    
    # Test with quotes in query
    results = await db.search_observations_fts('security "phrase" test', project="proj")
    # Should not crash and should find nothing (or handles quotes safely)
    assert isinstance(results, list)

@pytest.mark.asyncio
async def test_privacy_case_count_bypass():
    content = "<PRIVATE>Secret</PRIVATE>"
    # Previously this returned 0 because it was case-sensitive
    count = _count_tags(content)
    assert count == 1
    
    cleaned, stripped = strip_memory_tags(content)
    assert stripped is True
    assert "Secret" not in cleaned

@pytest.mark.asyncio
async def test_tag_clause_column_validation(db):
    params = []
    # Test internal _tag_clause with invalid column
    clause = db._tag_clause(["tag1"], params, column="DROP TABLE sessions; --")
    # Should return None or handle safely (never include untrusted column name)
    assert clause is None or "DROP TABLE" not in clause

@pytest.mark.asyncio
async def test_error_sanitization():
    from fastapi import HTTPException
    from ai_mem.server import add_memory
    from pydantic import BaseModel
    
    class FakeMem(BaseModel):
        content: str
        obs_type: str = "note"
    
    # Mocking manager to throw a specific error with sensitive info
    with mock.patch("ai_mem.server.get_manager") as mock_manager:
        mock_manager.return_value.add_observation.side_effect = Exception("SENSITIVE INFO: Database password leaked")
        
        request = mock.Mock()
        request.headers = {}
        
        with pytest.raises(HTTPException) as exc:
            await add_memory(FakeMem(content="test"), request)
        
        assert exc.value.status_code == 500
        assert "SENSITIVE INFO" not in str(exc.value.detail)
        assert exc.value.detail == "Internal server error"
