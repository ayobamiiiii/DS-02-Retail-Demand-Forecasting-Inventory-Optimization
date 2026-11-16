import unittest
import pandas as pd

from src.forecasting.hierarchy import rollup_favorita


class FavoritaHierarchyTests(unittest.TestCase):
    def test_bottom_up_rollups_include_item_store_city_state_and_overall(self):
        bottom=pd.DataFrame([
            {"date":"2020-01-01","id":"1_10","item_nbr":10,"store_nbr":1,"forecast":3.0,"lower":2.0,"upper":4.0},
            {"date":"2020-01-01","id":"1_11","item_nbr":11,"store_nbr":1,"forecast":7.0,"lower":5.0,"upper":9.0},
            {"date":"2020-01-01","id":"2_10","item_nbr":10,"store_nbr":2,"forecast":5.0,"lower":4.0,"upper":6.0},
        ])
        stores=pd.DataFrame([
            {"store_nbr":1,"city":"Quito","state":"Pichincha"},
            {"store_nbr":2,"city":"Quito","state":"Pichincha"},
        ])
        result=rollup_favorita(bottom,stores)
        self.assertEqual(set(result.level),{"item_store","item","store","city","state","overall"})
        self.assertAlmostEqual(result[(result.level=="city")].forecast.sum(),15.0)
        self.assertAlmostEqual(result[(result.level=="state")].forecast.sum(),15.0)
        self.assertAlmostEqual(result[(result.level=="overall")].forecast.sum(),15.0)
        self.assertAlmostEqual(result[(result.level=="item")].forecast.sum(),15.0)


if __name__=="__main__": unittest.main()
