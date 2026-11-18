import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.favorita import choose_items
from src.evaluation.model_comparison import _choose_common_ids
from src.pipeline import selection_cutoff_from_test_start


class SampleSelectionLeakageTests(unittest.TestCase):
    def test_item_ranking_ignores_sales_on_and_after_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            pd.DataFrame([
                {"date": "2020-01-01", "item_nbr": 10, "unit_sales": 5},
                {"date": "2020-01-01", "item_nbr": 20, "unit_sales": 4},
                {"date": "2020-01-02", "item_nbr": 20, "unit_sales": 1000},
            ]).to_csv(path, index=False)
            selected = choose_items(path, n_items=1, chunksize=1, before_date="2020-01-02")
            self.assertEqual(selected, [10])

    def test_common_series_ranking_ignores_future_sales(self):
        panel = pd.DataFrame([
            {"date": "2020-01-01", "store_nbr": 1, "id": "1_10", "unit_sales": 5},
            {"date": "2020-01-01", "store_nbr": 1, "id": "1_20", "unit_sales": 4},
            {"date": "2020-01-02", "store_nbr": 1, "id": "1_20", "unit_sales": 1000},
        ])
        panel["date"] = pd.to_datetime(panel.date)
        selected = _choose_common_ids(panel, limit=1, selection_cutoff="2020-01-02")
        self.assertEqual(selected, ["1_10"])

    def test_item_ranking_requires_explicit_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            pd.DataFrame([{"date": "2020-01-01", "item_nbr": 10, "unit_sales": 5}]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "before_date is required"):
                choose_items(path)

    def test_pipeline_cutoff_precedes_earliest_selection_or_validation_window(self):
        config = {
            "validation_days": 28,
            "model_comparison": {"evaluation_days": 28, "calibration_days": 14, "selection_days": 14},
        }
        cutoff = selection_cutoff_from_test_start("2017-08-16", config)
        self.assertEqual(cutoff, pd.Timestamp("2017-06-21"))


if __name__ == "__main__":
    unittest.main()
