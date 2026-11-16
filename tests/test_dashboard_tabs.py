"""Smoke checks for the Dash entry point and its six registered tab views."""
import unittest

from dashboard.app import app, forecast, items, stores, update


class DashboardTabTests(unittest.TestCase):
    def test_home_route_loads(self):
        response = app.server.test_client().get("/")
        self.assertEqual(response.status_code, 200)

    def test_all_six_tab_callbacks_return_content(self):
        start = forecast.date.min().date().isoformat() if not forecast.empty else None
        end = forecast.date.max().date().isoformat() if not forecast.empty else None
        for tab in (
            "Executive Overview", "Sales Trends", "Demand Forecasts",
            "Model Performance", "Inventory Optimization", "Store and Item Analysis",
        ):
            with self.subTest(tab=tab):
                content = update(stores, items, start, end, tab)
                self.assertIsNotNone(content)
                self.assertTrue(content.children)


if __name__ == "__main__":
    unittest.main()
