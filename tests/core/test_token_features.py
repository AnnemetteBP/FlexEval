import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flex_moe_toolkit.utils.token_features import align_offsets_to_words, fragmentation_bucket, whitespace_word_spans


class TokenFeatureUtilsTests(unittest.TestCase):
    def test_whitespace_word_spans_preserves_word_boundaries(self):
        spans = whitespace_word_spans("Hej verden  igen")
        self.assertEqual([(item.index, item.text, item.start, item.end) for item in spans], [
            (0, "Hej", 0, 3),
            (1, "verden", 4, 10),
            (2, "igen", 12, 16),
        ])

    def test_fragmentation_bucket_groups_high_fragment_words(self):
        self.assertEqual(fragmentation_bucket(None), "unknown")
        self.assertEqual(fragmentation_bucket(0), "unknown")
        self.assertEqual(fragmentation_bucket(1), "1")
        self.assertEqual(fragmentation_bucket(2), "2")
        self.assertEqual(fragmentation_bucket(3), "3+")
        self.assertEqual(fragmentation_bucket(8), "3+")

    def test_align_offsets_to_words_counts_subtokens_per_word(self):
        text = "alpha beta"
        offsets = [(0, 2), (2, 5), (0, 0), (6, 10)]
        rows = align_offsets_to_words(text, offsets)

        self.assertEqual(rows[0]["word_idx"], 0)
        self.assertEqual(rows[0]["word_text"], "alpha")
        self.assertEqual(rows[0]["word_subtoken_count"], 2)
        self.assertEqual(rows[0]["fragmentation_bucket"], "2")

        self.assertEqual(rows[1]["word_idx"], 0)
        self.assertEqual(rows[1]["word_subtoken_count"], 2)

        self.assertIsNone(rows[2]["word_idx"])
        self.assertEqual(rows[2]["fragmentation_bucket"], "unknown")

        self.assertEqual(rows[3]["word_idx"], 1)
        self.assertEqual(rows[3]["word_text"], "beta")
        self.assertEqual(rows[3]["word_subtoken_count"], 1)
        self.assertEqual(rows[3]["fragmentation_bucket"], "1")


if __name__ == "__main__":
    unittest.main()
