"""Smoke checks for the Dash entry point and its six registered tab views."""
import unittest

import pandas as pd
from dash import dcc, dash_table

from dashboard.app import (
    app, forecast, history, items, stores, update, update_store_item_summary,
)


def walk_components(component):
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from walk_components(child)
    elif hasattr(component, "to_plotly_json"):
        yield component
        yield from walk_components(component.to_plotly_json()["props"].get("children"))


def graph_components(component):
    return [node for node in walk_components(component) if isinstance(node, dcc.Graph)]


def leaf_text(component):
    if isinstance(component, (list, tuple)):
        return [text for child in component for text in leaf_text(child)]
    if hasattr(component, "to_plotly_json"):
        return leaf_text(component.to_plotly_json()["props"].get("children"))
    if isinstance(component, str):
        return [component]
    return []


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

    def test_sales_trends_uses_actual_history_only(self):
        start = history.date.min().date().isoformat()
        end = history.date.max().date().isoformat()
        content = update(stores, items, start, end, "Sales Trends")
        graphs = graph_components(content)
        self.assertEqual([g.figure.layout.title.text for g in graphs], [
            "Daily historical sales · actuals only",
            "Average actual sales by weekday",
            "Historical sales by selected store",
        ])
        trend, weekday, store_compare = [g.figure for g in graphs]
        expected = history.groupby("date").sales.sum().sort_index()
        self.assertEqual(list(trend.data[0].y), list(expected.to_numpy()))
        self.assertEqual(list(weekday.data[0].x), ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        self.assertEqual({t.name for t in store_compare.data}, {str(s) for s in stores})
        self.assertNotIn("forecast", " ".join(t.name.lower() for g in graphs for t in g.figure.data))
        self.assertNotIn("interval", " ".join(g.figure.layout.title.text.lower() for g in graphs))

    def test_sales_trend_kpis_obey_global_store_item_and_date_filters(self):
        store = stores[0]
        item = items[0]
        start, end = "2017-01-01", "2017-01-07"
        expected = history.loc[
            history.store_nbr.eq(store) & history.item_nbr.eq(item) & history.date.between(start, end), "sales"
        ].sum()
        content = update([store], [item], start, end, "Sales Trends")
        self.assertIn(f"{expected:,.0f}", leaf_text(content))
        store_fig = graph_components(content)[2].figure
        self.assertEqual({trace.name for trace in store_fig.data}, {str(store)})

    def test_demand_forecasts_distinguish_actual_validation_and_future_predictions(self):
        start = history.date.min().date().isoformat()
        end = forecast.date.max().date().isoformat()
        content = update(stores, items, start, end, "Demand Forecasts")
        graphs = graph_components(content)
        self.assertEqual(len(graphs), 1)
        figure = graphs[0].figure
        names = {trace.name for trace in figure.data}
        self.assertIn("Historical actual sales", names)
        self.assertIn("Validation actual sales", names)
        self.assertIn("Selected-model validation prediction", names)
        self.assertIn("90% prediction interval", names)
        self.assertIn("Future model forecast", names)
        forecast_trace = next(trace for trace in figure.data if trace.name == "Future model forecast")
        self.assertEqual(pd.to_datetime(forecast_trace.x).min(), forecast.date.min())
        self.assertEqual(pd.to_datetime(forecast_trace.x).max(), forecast.date.max())

    def test_executive_overview_keeps_kpis_without_detailed_forecast_chart(self):
        start = history.date.min().date().isoformat()
        end = forecast.date.max().date().isoformat()
        content = update(stores, items, start, end, "Executive Overview")
        self.assertEqual(graph_components(content), [])
        texts = leaf_text(content)
        self.assertIn("Forecast units", texts)
        self.assertIn("Recommended order units", texts)
        self.assertIn("Modeled inventory cost", texts)

    def test_store_item_analysis_is_searchable_sortable_and_formatted(self):
        start = forecast.date.min().date().isoformat()
        end = forecast.date.max().date().isoformat()
        content = update(stores, items, start, end, "Store and Item Analysis")
        tables = [node for node in walk_components(content) if isinstance(node, dash_table.DataTable)]
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual([column["name"] for column in table.columns], [
            "Store Number", "Item Number", "Forecast Units",
        ])
        self.assertEqual(table.sort_action, "native")
        self.assertEqual(table.sort_mode, "multi")
        self.assertEqual(table.filter_action, "native")
        self.assertEqual(table.filter_options["case_sensitive"], False)
        self.assertEqual(table.columns[2]["format"].to_plotly_json()["specifier"], ",.2f")

        source = forecast.loc[forecast.date.between(start, end)]
        expected = source.groupby(["store_nbr", "item_nbr"]).forecast.sum().sort_index()
        observed = pd.DataFrame(table.data).set_index(["store_nbr", "item_nbr"]).forecast_units.sort_index()
        pd.testing.assert_series_equal(observed, expected.rename("forecast_units"), check_names=True)
        self.assertIn(f"{expected.sum():,.2f}", leaf_text(content))
        self.assertIn(f"{source.store_nbr.nunique():,}", leaf_text(content))
        self.assertIn(f"{source.item_nbr.nunique():,}", leaf_text(content))

    def test_store_item_table_and_kpis_obey_global_filters_and_table_search(self):
        row = forecast.iloc[0]
        store, item = row.store_nbr, row.item_nbr
        start = row.date.date().isoformat()
        content = update([store], [item], start, start, "Store and Item Analysis")
        tables = [node for node in walk_components(content) if isinstance(node, dash_table.DataTable)]
        self.assertEqual(len(tables), 1)
        data = tables[0].data
        self.assertTrue(data)
        self.assertEqual({record["store_nbr"] for record in data}, {store})
        self.assertEqual({record["item_nbr"] for record in data}, {item})
        expected = forecast.loc[
            forecast.store_nbr.eq(store) & forecast.item_nbr.eq(item) & forecast.date.eq(row.date), "forecast"
        ].sum()
        self.assertEqual(data[0]["forecast_units"], expected)

        # Dash's derived_virtual_data is the filtered result set after native column search.
        filtered = data[:1]
        self.assertEqual(update_store_item_summary(filtered), (f"{expected:,.2f}", "1", "1"))
        self.assertEqual(update_store_item_summary([]), ("0.00", "0", "0"))


if __name__ == "__main__":
    unittest.main()
