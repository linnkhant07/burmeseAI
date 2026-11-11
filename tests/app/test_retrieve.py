"""Retrieval tests. The fixture is project-written and is not a scraped page."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from burmeseai.retrieve import (
    NGRAM_WIDTHS,
    SCORE_CUTOFF,
    Chunk,
    Index,
    content_runs,
    grapheme_clusters,
    load_chunks,
    ngrams,
)

FIXTURE = Path(__file__).with_name("safety_fixture.json")


class GraphemeTests(unittest.TestCase):
    def test_widths_are_three_four_and_five(self) -> None:
        self.assertEqual(NGRAM_WIDTHS, (3, 4, 5))

    def test_overlapping_latin_ngrams(self) -> None:
        self.assertEqual(
            ngrams("abcdef"),
            [
                "abc",
                "bcd",
                "cde",
                "def",
                "abcd",
                "bcde",
                "cdef",
                "abcde",
                "bcdef",
            ],
        )

    def test_bigrams_are_absent(self) -> None:
        self.assertEqual(ngrams("ab"), [])
        self.assertNotIn("ab", ngrams("abcd"))

    def test_punctuation_and_spaces_split_runs(self) -> None:
        self.assertEqual(ngrams("abc.def"), ["abc", "def"])
        self.assertEqual(ngrams("ab.cd"), [])
        self.assertNotIn("atw", ngrams("heat wave"))
        self.assertIn("heat", ngrams("heat wave"))

    def test_casefold_applies_to_ngrams_only(self) -> None:
        self.assertEqual(grapheme_clusters("AbC"), ["A", "b", "C"])
        self.assertEqual(ngrams("Heat"), ngrams("heat"))

    def test_combining_mark_stays_on_its_base(self) -> None:
        self.assertEqual(grapheme_clusters("e\u0301"), ["é"])
        self.assertEqual(grapheme_clusters("ကာ"), ["ကာ"])
        self.assertEqual(grapheme_clusters("င်"), ["င်"])
        self.assertEqual(grapheme_clusters("လျ"), ["လျ"])

    def test_virama_stack_is_one_cluster_and_asat_is_not(self) -> None:
        stacked = "က\u1039ခ"
        self.assertEqual(grapheme_clusters(stacked), [stacked])
        self.assertEqual(grapheme_clusters("က\u103aခ"), ["က\u103a", "ခ"])
        self.assertEqual(grapheme_clusters("ကခ"), ["က", "ခ"])

    def test_myanmar_word_and_its_single_trigram(self) -> None:
        self.assertEqual(grapheme_clusters("မြန်မာ"), ["မြ", "န်", "မာ"])
        self.assertEqual(ngrams("မြန်မာ"), ["မြန်မာ"])
        self.assertEqual(grapheme_clusters("ငလျင်"), ["င", "လျ", "င်"])
        self.assertEqual(ngrams("ငလျင်"), ["ငလျင်"])

    def test_hangul_and_regional_indicators(self) -> None:
        self.assertEqual(grapheme_clusters("\u1100\u1161"), ["\uac00"])
        self.assertEqual(grapheme_clusters("\u1100\u1100"), ["\u1100\u1100"])
        self.assertEqual(grapheme_clusters("한"), ["한"])
        pair = "\U0001F1F2\U0001F1F2"
        self.assertEqual(grapheme_clusters(pair), [pair])
        self.assertEqual(grapheme_clusters(pair + "\U0001F1E6"), [pair, "\U0001F1E6"])

    def test_carriage_return_line_feed_stays_together(self) -> None:
        self.assertEqual(grapheme_clusters("a\r\nb"), ["a", "\r\n", "b"])


class IndexTests(unittest.TestCase):
    def test_empty_index_refuses_without_hits(self) -> None:
        result = Index().search("earthquake nearby")
        self.assertTrue(result.refused)
        self.assertEqual(result.reason, "empty_index")
        self.assertEqual(result.hits, [])
        self.assertIsNone(result.top_score)

    def test_short_query_has_no_ngram_and_scores_low(self) -> None:
        index = Index([Chunk(id="c", text="abcd", license="CC0-1.0")])
        result = index.search("ab")
        self.assertTrue(result.refused)
        self.assertEqual(result.reason, "low_score")
        self.assertEqual(result.hits, [])
        self.assertIsNotNone(result.top_score)
        self.assertLess(result.top_score, SCORE_CUTOFF)

    def test_five_grapheme_exact_match_clears_the_cutoff(self) -> None:
        index = Index([Chunk(id="c", text="abcde", license="CC0-1.0")])
        result = index.search("abcde")
        self.assertFalse(result.refused)
        self.assertIsNone(result.reason)
        self.assertEqual([hit.chunk_id for hit in result.hits], ["c"])
        self.assertGreaterEqual(result.top_score, SCORE_CUTOFF)
        self.assertEqual(result.hits[0].text, "abcde")

    def test_four_grapheme_exact_match_stays_below_the_cutoff(self) -> None:
        index = Index([Chunk(id="c", text="abcd", license="CC0-1.0")])
        result = index.search("abcd")
        self.assertTrue(result.refused)
        self.assertEqual(result.reason, "low_score")
        self.assertEqual(result.hits, [])
        self.assertGreater(result.top_score, 0.0)
        self.assertLess(result.top_score, SCORE_CUTOFF)

    def test_missing_license_and_duplicate_id_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Chunk(id="c", text="abcd", license="  ")
        index = Index([Chunk(id="c", text="abcd", license="CC0-1.0")])
        with self.assertRaises(ValueError):
            index.add(Chunk(id="c", text="abcd", license="CC0-1.0"))

    def test_top_k_limits_hits(self) -> None:
        chunks = [
            Chunk(id=f"c{i}", text=f"unique{i} sharedtoken", license="CC0-1.0")
            for i in range(6)
        ]
        index = Index(chunks)
        result = index.search("sharedtoken", top_k=5)
        self.assertFalse(result.refused)
        self.assertEqual(len(result.hits), 5)

    def test_fixture_file_is_project_written(self) -> None:
        raw = FIXTURE.read_text(encoding="utf-8")
        self.assertNotIn("http://", raw)
        self.assertNotIn("https://", raw)
        self.assertNotIn("<html", raw.lower())
        payload = json.loads(raw)
        self.assertGreaterEqual(len(payload), 3)
        topics = set()
        for item in payload:
            self.assertEqual(item["source"], "project-written")
            self.assertIn("CC0-1.0", item["license"])
            self.assertNotIn("external page", item["text"].lower())
            topics.add(item["topic"])
            lowered = item["text"].lower()
            for banned in ("drink water", "drop, cover", "cpr", "dosage", "first aid"):
                self.assertNotIn(banned, lowered)
        self.assertEqual(topics, {"earthquake", "heat", "health"})

    def test_fixture_retrieval_hits_the_named_chunk(self) -> None:
        index = Index(load_chunks(str(FIXTURE)))
        earthquake = index.search("earthquake")
        self.assertFalse(earthquake.refused)
        self.assertEqual([hit.chunk_id for hit in earthquake.hits], ["fixture-earthquake-1"])
        heat = index.search("heat")
        self.assertEqual([hit.chunk_id for hit in heat.hits], ["fixture-heat-1"])
        health = index.search("health")
        self.assertEqual([hit.chunk_id for hit in health.hits], ["fixture-health-1"])
        myanmar = index.search("ငလျင်")
        self.assertEqual([hit.chunk_id for hit in myanmar.hits], ["fixture-earthquake-my-1"])

    def test_common_trigram_scores_below_the_cutoff(self) -> None:
        index = Index(load_chunks(str(FIXTURE)))
        result = index.search("not")
        self.assertTrue(result.refused)
        self.assertEqual(result.reason, "low_score")
        self.assertEqual(result.hits, [])
        self.assertGreater(result.top_score, 0.0)
        self.assertLess(result.top_score, SCORE_CUTOFF)

    def test_fixture_miss_is_a_low_score_with_no_quoted_text(self) -> None:
        index = Index(load_chunks(str(FIXTURE)))
        result = index.search("aftershock")
        self.assertTrue(result.refused)
        self.assertEqual(result.reason, "low_score")
        self.assertEqual(result.hits, [])
        self.assertLess(result.top_score, SCORE_CUTOFF)
        self.assertEqual(result.top_score, 0.0)

    def test_load_chunks_rejects_a_bad_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('[{"id": "c", "text": "abcd"}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_chunks(str(path))

    def test_content_runs_drop_the_hyphen(self) -> None:
        runs = content_runs("heat-wave")
        self.assertEqual(runs, [["h", "e", "a", "t"], ["w", "a", "v", "e"]])


if __name__ == "__main__":
    unittest.main()
