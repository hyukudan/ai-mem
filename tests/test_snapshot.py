import json
import tempfile
import unittest

from ai_mem.cli import _infer_format, _read_export_file


class SnapshotFileTests(unittest.TestCase):
    def test_read_jsonl_snapshot_with_header(self):
        header = {"snapshot": {"snapshot_id": "snap-123", "since": "2024-01-01"}}
        observation = {"id": "obs-1", "content": "hello", "created_at": 100.0}
        with tempfile.NamedTemporaryFile("w+", delete=False) as handle:
            handle.write(json.dumps(header) + "\n")
            handle.write(json.dumps(observation) + "\n")
            path = handle.name

        fmt = _infer_format(path, None, "jsonl")
        items, metadata = _read_export_file(path, fmt)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.get("snapshot_id"), "snap-123")
        self.assertEqual(len(items), 1)

    def test_read_json_snapshot_with_observations_list(self):
        payload = {
            "snapshot": {"snapshot_id": "snap-456"},
            "observations": [
                {"id": "obs-2", "content": "world", "created_at": 200.0},
            ],
        }
        with tempfile.NamedTemporaryFile("w+", delete=False) as handle:
            json.dump(payload, handle)
            path = handle.name

        fmt = _infer_format(path, None, "json")
        items, metadata = _read_export_file(path, fmt)
        self.assertEqual(metadata.get("snapshot_id"), "snap-456")
        self.assertEqual(items[0]["id"], "obs-2")

