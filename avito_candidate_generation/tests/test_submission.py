from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.submission import build_submission, validate_submission, write_submission


Q1 = "00WuFMeXSFZBxSzT"
Q2 = "03ztb1gtRFC4K4vP"
I1 = "1382564bf8994a83"
I2 = "121fa7f5e765ce00"


class SubmissionTest(unittest.TestCase):
    def test_build_and_roundtrip(self) -> None:
        frame = build_submission([Q1, Q2], {Q1: [I1, I1, I2], Q2: [I2]}, {I1, I2})
        self.assertEqual(frame.loc[0, "answer"], f"{I1} {I2}")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answer.csv"
            write_submission(frame, output, [Q1, Q2], {I1, I2})
            reread = pd.read_csv(output, dtype=str)
            self.assertEqual(list(reread.columns), ["query_id", "answer"])

    def test_unknown_item_rejected(self) -> None:
        frame = pd.DataFrame({"query_id": [Q1], "answer": ["ffffffffffffffff"]})
        with self.assertRaises(ValueError):
            validate_submission(frame, [Q1], {I1})

    def test_invalid_id_format_rejected(self) -> None:
        frame = pd.DataFrame({"query_id": [Q1], "answer": ["ABC"]})
        with self.assertRaises(ValueError):
            validate_submission(frame, [Q1], {I1})

    def test_extra_query_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_submission([Q1], {Q1: [I1], Q2: [I2]}, {I1, I2})


if __name__ == "__main__":
    unittest.main()
