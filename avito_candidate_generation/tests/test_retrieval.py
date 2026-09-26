from __future__ import annotations

import unittest

import pandas as pd

from src.pipeline import run_sparse_baseline
from src.validation import make_query_signature, split_by_query_signature


class RetrievalSmokeTest(unittest.TestCase):
    def test_sparse_pipeline_returns_unique_candidates(self) -> None:
        queries = pd.DataFrame([
            {
                "query_id": "0123456789ABCDEF",
                "search_query": "ремонт телевизора",
                "search_infn_params_text": "",
                "search_category": "10",
                "search_location_id": "77",
                "search_is_delivery_search": 0,
            }
        ])
        items = pd.DataFrame([
            {
                "item_id": "1111111111111111",
                "item_title_raw": "Ремонт телевизоров",
                "item_infn_params_text": "мастер",
                "item_description_raw": "быстрый ремонт телевизора",
                "item_category_id": "10",
                "item_microcat_id": "100",
                "item_location_id": "77",
            },
            {
                "item_id": "2222222222222222",
                "item_title_raw": "Монтаж дверей",
                "item_infn_params_text": "",
                "item_description_raw": "установка входных дверей",
                "item_category_id": "20",
                "item_microcat_id": "200",
                "item_location_id": "78",
            },
        ])
        config = {
            "batch_size": 1,
            "final_top_k": 2,
            "fusion": {"mode": "rrf", "rrf_k": 60, "retrieval_top_k": 2},
            "word_tfidf": {"ngram_range": [1, 2], "min_df": 1},
            "char_tfidf": {"ngram_range": [3, 5], "min_df": 1},
            "text": {"title_weight": 2, "params_weight": 1, "description_weight": 1},
            "metadata": {"enabled": True, "category_boost": 0.05, "location_boost": 0.02},
        }
        result = run_sparse_baseline(queries, items, config)["0123456789ABCDEF"]
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result), len(set(result)))
        self.assertEqual(result[0], "1111111111111111")

        config["scoped_retrieval"] = {
            "enabled": True,
            "history_quota": 1,
            "microcategory_quota": 1,
            "location_quota": 1,
            "global_quota": 1,
        }
        scoped = run_sparse_baseline(
            queries,
            items,
            config,
            history_candidates={"0123456789ABCDEF": ["1111111111111111"]},
            predicted_microcats={"0123456789ABCDEF": ["100"]},
        )["0123456789ABCDEF"]
        self.assertEqual(scoped[0], "1111111111111111")
        self.assertEqual(len(scoped), len(set(scoped)))

    def test_signature_split_has_no_overlap(self) -> None:
        rows = []
        for index in range(10):
            rows.append({
                "search_query": f"query {index}",
                "search_location_id": str(index),
                "search_is_delivery_search": index % 2,
                "search_infn_params_text": None,
                "search_category": "service",
                "item_id": f"{index:016x}",
            })
        frame = pd.DataFrame(rows)
        left, right = split_by_query_signature(frame, validation_size=0.3, random_state=42)
        self.assertFalse(set(make_query_signature(left)) & set(make_query_signature(right)))


if __name__ == "__main__":
    unittest.main()
