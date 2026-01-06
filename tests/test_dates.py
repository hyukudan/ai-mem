import time
import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ai_mem.memory import _parse_date


class DateParseTests(unittest.TestCase):
    def test_relative_dates(self):
        now = time.time()
        parsed = _parse_date("2h")
        self.assertIsNotNone(parsed)
        delta = now - float(parsed)
        self.assertGreater(delta, 2 * 3600 - 10)
        self.assertLess(delta, 2 * 3600 + 10)
