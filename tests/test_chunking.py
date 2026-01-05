import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ai_mem.chunking import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_chunk_overlap(self):
        chunks = chunk_text("abcdefghij", chunk_size=4, chunk_overlap=1)
        self.assertEqual(chunks, ["abcd", "defg", "ghij"])


if __name__ == "__main__":
    unittest.main()
