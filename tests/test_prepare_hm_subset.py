from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.prepare_hm_subset import output_paths, prepare_articles, prepare_customers, prepare_transactions


class PrepareHmSubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

        self.articles_path = self.base_path / "articles.csv"
        self.customers_path = self.base_path / "customers.csv"
        self.transactions_path = self.base_path / "transactions.csv"
        self.images_dir = self.base_path / "images"
        (self.images_dir / "000").mkdir(parents=True, exist_ok=True)
        (self.images_dir / "000" / "0000000001.jpg").write_bytes(b"fake-image")

        pd.DataFrame(
            [
                {"article_id": "0000000001", "prod_name": "Dress", "product_type_name": "Dress", "product_group_name": "Garment Full body", "graphical_appearance_name": "Solid", "colour_group_name": "Black", "perceived_colour_value_name": "Dark", "perceived_colour_master_name": "Black", "department_name": "Dresses", "index_name": "Ladieswear", "section_name": "Contemporary", "garment_group_name": "Dresses", "detail_desc": "Black summer dress"},
                {"article_id": "0000000002", "prod_name": "Jeans", "product_type_name": "Trousers", "product_group_name": "Garment Lower body", "graphical_appearance_name": "Solid", "colour_group_name": "Blue", "perceived_colour_value_name": "Medium", "perceived_colour_master_name": "Blue", "department_name": "Trousers", "index_name": "Ladieswear", "section_name": "Denim", "garment_group_name": "Trousers", "detail_desc": "Blue wide-leg jeans"},
            ]
        ).to_csv(self.articles_path, index=False)

        pd.DataFrame(
            [
                {"customer_id": "c1", "FN": None, "Active": None, "club_member_status": None, "fashion_news_frequency": None, "age": None, "postal_code": None},
                {"customer_id": "c2", "FN": 1, "Active": 1, "club_member_status": "ACTIVE", "fashion_news_frequency": "Regularly", "age": 24, "postal_code": "12345"},
            ]
        ).to_csv(self.customers_path, index=False)

        pd.DataFrame(
            [
                {"customer_id": "c1", "article_id": "1", "t_dat": "2020-09-20"},
                {"customer_id": "c2", "article_id": "2", "t_dat": "2020-09-10"},
            ]
        ).to_csv(self.transactions_path, index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prepare_transactions_subset_mode(self) -> None:
        filtered = prepare_transactions(self.transactions_path, days=5)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["customer_id"], "c1")
        self.assertEqual(filtered.iloc[0]["article_id"], "0000000001")

    def test_prepare_transactions_full_mode(self) -> None:
        filtered = prepare_transactions(self.transactions_path, days=0)
        self.assertEqual(len(filtered), 2)

    def test_prepare_articles_and_customers_fill_defaults(self) -> None:
        articles = prepare_articles(self.articles_path, {"0000000001"}, images_dir=self.images_dir)
        customers = prepare_customers(self.customers_path, {"c1"})

        self.assertEqual(len(articles), 1)
        self.assertIn("text_for_embedding", articles.columns)
        self.assertEqual(customers.iloc[0]["club_member_status"], "UNKNOWN")
        self.assertEqual(customers.iloc[0]["age"], -1)
        self.assertTrue(bool(articles.iloc[0]["image_available"]))
        self.assertEqual(articles.iloc[0]["image_relative_path"], "000/0000000001.jpg")

    def test_prepare_articles_without_images_marks_unavailable(self) -> None:
        articles = prepare_articles(self.articles_path, {"0000000002"}, images_dir=self.images_dir)
        self.assertFalse(bool(articles.iloc[0]["image_available"]))
        self.assertEqual(articles.iloc[0]["image_local_path"], "")

    def test_output_paths_with_prefix(self) -> None:
        articles_output, customers_output, transactions_output = output_paths(self.base_path, "full_")
        self.assertEqual(articles_output.name, "full_articles_cleaned.csv")
        self.assertEqual(customers_output.name, "full_customers_cleaned.csv")
        self.assertEqual(transactions_output.name, "full_transactions_cleaned.csv")


if __name__ == "__main__":
    unittest.main()
