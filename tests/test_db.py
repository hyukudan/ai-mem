import tempfile
import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ai_mem.db import DatabaseManager
from ai_mem.models import Observation, Session


class DatabaseTests(unittest.TestCase):
    def test_fts_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            session = Session(project="proj")
            db.add_session(session)

            obs = Observation(
                session_id=session.id,
                project="proj",
                type="note",
                content="Hello world from ai-mem",
                summary="Hello world",
            )
            db.add_observation(obs)

            results = db.search_observations_fts("Hello", project="proj", limit=5)
            self.assertTrue(results)
            self.assertEqual(results[0].id, obs.id)

    def test_tag_filters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            session = Session(project="proj")
            db.add_session(session)

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
            db.add_observation(obs_alpha)
            db.add_observation(obs_beta)

            results = db.search_observations_fts(
                "note",
                project="proj",
                tag_filters=["alpha"],
                limit=5,
            )
            self.assertEqual([item.id for item in results], [obs_alpha.id])

            recent = db.get_recent_observations(
                "proj",
                limit=5,
                tag_filters=["beta"],
            )
            self.assertEqual([item.id for item in recent], [obs_beta.id])

            stats = db.get_stats(project="proj", tag_filters=["alpha"])
            self.assertEqual(stats["total"], 1)

    def test_update_observation_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            session = Session(project="proj")
            db.add_session(session)

            obs = Observation(
                session_id=session.id,
                project="proj",
                type="note",
                content="Tagged note",
                summary="Tagged note",
                tags=["alpha"],
            )
            db.add_observation(obs)

            updated = db.update_observation_tags(obs.id, ["alpha", "beta"])
            self.assertEqual(updated, 1)
            stored = db.get_observation(obs.id)
            self.assertEqual(stored["tags"], ["alpha", "beta"])

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            session_a = Session(project="proj-a")
            session_b = Session(project="proj-b")
            db.add_session(session_a)
            db.add_session(session_b)

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
                db.add_observation(obs)

            stats_all = db.get_stats()
            self.assertEqual(stats_all["total"], 3)
            self.assertEqual(stats_all["by_project"][0]["project"], "proj-a")
            self.assertEqual(stats_all["by_project"][0]["count"], 2)

            stats_a = db.get_stats(project="proj-a")
            self.assertEqual(stats_a["total"], 2)
            self.assertEqual(stats_a["by_project"][0]["project"], "proj-a")
            self.assertEqual(stats_a["by_project"][0]["count"], 2)
            self.assertEqual(stats_a["top_tags"][0]["tag"], "shared")
            self.assertEqual(stats_a["top_tags"][0]["count"], 2)

    def test_stats_date_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            session = Session(project="proj")
            db.add_session(session)

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
            db.add_observation(older)
            db.add_observation(newer)

            stats = db.get_stats(date_start=150.0)
            self.assertEqual(stats["total"], 1)
            self.assertEqual(stats["by_project"][0]["project"], "proj")
            self.assertEqual(stats["by_project"][0]["count"], 1)
            self.assertEqual(stats["by_day"][0]["count"], 1)
            self.assertEqual(stats["recent_total"], 1)

    def test_stats_by_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            session = Session(project="proj")
            db.add_session(session)

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
            db.add_observation(obs_early)
            db.add_observation(obs_late)

            stats = db.get_stats(day_limit=5)
            day_counts = {item["day"]: item["count"] for item in stats["by_day"]}
            self.assertEqual(day_counts.get("1970-01-02"), 1)
            self.assertEqual(day_counts.get("1970-01-03"), 1)
            self.assertEqual(stats["recent_total"], 2)

    def test_stats_obs_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            session = Session(project="proj")
            db.add_session(session)

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
            db.add_observation(obs_note)
            db.add_observation(obs_bug)

            stats = db.get_stats(obs_type="bugfix")
            self.assertEqual(stats["total"], 1)
            self.assertEqual(stats["by_type"][0]["type"], "bugfix")
            self.assertEqual(stats["top_tags"][0]["tag"], "beta")

    def test_stats_top_tags_by_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            session = Session(project="proj")
            db.add_session(session)

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
            db.add_observation(obs_note)
            db.add_observation(obs_bug)

            stats = db.get_stats(type_tag_limit=1)
            grouped = {item["type"]: item["tags"][0]["tag"] for item in stats["top_tags_by_type"]}
            self.assertEqual(grouped.get("note"), "alpha")
            self.assertEqual(grouped.get("bugfix"), "beta")

    def test_before_after_filters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            session = Session(project="proj")
            db.add_session(session)

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
                db.add_observation(obs)

            before_notes = db.get_observations_before(
                project="proj",
                anchor_time=250.0,
                limit=10,
                obs_type="note",
            )
            self.assertEqual([item.id for item in before_notes], [obs_early.id])

            after_notes = db.get_observations_after(
                project="proj",
                anchor_time=250.0,
                limit=10,
                obs_type="note",
            )
            self.assertEqual([item.id for item in after_notes], [obs_late.id])

            before_date = db.get_observations_before(
                project="proj",
                anchor_time=350.0,
                limit=10,
                date_start=250.0,
            )
            self.assertEqual([item.id for item in before_date], [obs_late.id])

            after_date = db.get_observations_after(
                project="proj",
                anchor_time=150.0,
                limit=10,
                date_end=250.0,
            )
            self.assertEqual([item.id for item in after_date], [obs_mid.id])

    def test_stats_trend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            session = Session(project="proj")
            db.add_session(session)

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
            db.add_observation(obs_prev)
            db.add_observation(obs_recent_a)
            db.add_observation(obs_recent_b)

            stats = db.get_stats(day_limit=1)
            self.assertEqual(stats["recent_total"], 2)
            self.assertEqual(stats["previous_total"], 1)
            self.assertEqual(stats["trend_delta"], 1)

    def test_sessions_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            active = Session(project="proj-a", goal="Alpha goal", start_time=100.0)
            ended = Session(project="proj-a", goal="Beta goal", start_time=200.0, end_time=123.0)
            db.add_session(active)
            db.add_session(ended)

            sessions_all = db.list_sessions(project="proj-a")
            self.assertEqual(len(sessions_all), 2)

            sessions_active = db.list_sessions(project="proj-a", active_only=True)
            self.assertEqual(len(sessions_active), 1)
            self.assertEqual(sessions_active[0]["id"], active.id)

            goal_match = db.list_sessions(project="proj-a", goal_query="alpha")
            self.assertEqual(len(goal_match), 1)
            self.assertEqual(goal_match[0]["id"], active.id)

            recent = db.list_sessions(project="proj-a", date_start=150.0)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0]["id"], ended.id)

    def test_observations_by_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
            session_a = Session(project="proj-a")
            session_b = Session(project="proj-a")
            db.add_session(session_a)
            db.add_session(session_b)

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
            db.add_observation(obs_a)
            db.add_observation(obs_b)

            results = db.list_observations(session_id=session_a.id)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["id"], obs_a.id)


if __name__ == "__main__":
    unittest.main()
