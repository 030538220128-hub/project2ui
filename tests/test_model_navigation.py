"""Regression coverage for model caches, LSA similarity, free-text search, and Streamlit pages."""

from pathlib import Path
import unittest

import streamlit as st
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
SIMILARITY = "pages/1_Company_Similarity.py"
CLASSIFIER = "pages/2_Recommend_Classifier.py"
OVERVIEW = "pages/3_Company_Overview.py"


class ModelNavigationTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_resource.clear()

    def check_navigation(self, pages):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        for page in pages:
            with self.subTest(page=page):
                app.switch_page(page).run()
                self.assertEqual([e.message for e in app.exception], [])
                self.assertEqual([e.value for e in app.error], [])
                if page == SIMILARITY:
                    app.button[0].click().run()
                    self.assertEqual([e.message for e in app.exception], [])
                    self.assertEqual([e.value for e in app.error], [])
                    self.assertEqual(len(app.metric), 5)
                elif page == CLASSIFIER:
                    app.radio[0].set_value("✍️ Nhập review mới").run()
                    app.text_input[0].set_value("Great team and learning opportunities")
                    app.button[0].click().run()
                    self.assertEqual([e.message for e in app.exception], [])
                    self.assertEqual([e.value for e in app.error], [])
                    self.assertEqual(len(app.metric), 2)
                    self.assertEqual(app.metric[0].label, "Model Prediction")
                    self.assertIn(app.metric[0].value, ("Yes", "No"))
                elif page == OVERVIEW:
                    self.assertEqual([e.message for e in app.exception], [])
                    self.assertEqual([e.value for e in app.error], [])
                    self.assertGreaterEqual(len(app.metric), 3)

    def test_similarity_then_classifier_then_overview(self):
        self.check_navigation([SIMILARITY, CLASSIFIER, OVERVIEW])

    def test_overview_then_classifier_then_similarity(self):
        self.check_navigation([OVERVIEW, CLASSIFIER, SIMILARITY])

    def test_similarity_lsa_and_keyword_search(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        app.switch_page(SIMILARITY).run()

        # Test Method D (LSA)
        app.radio[1].set_value("D").run()
        app.button[0].click().run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual(len(app.metric), 5)

        # Test free-text keyword search
        app.radio[0].set_value("🔎 Tìm kiếm theo từ khóa / mô tả tự do").run()
        app.button[0].click().run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual(len(app.metric), 5)


if __name__ == "__main__":
    unittest.main()
