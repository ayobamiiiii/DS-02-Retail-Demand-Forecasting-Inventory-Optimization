import unittest
import pandas as pd

from src.data.favorita import add_lagged_covariates
from src.pipeline import _future_exog


class FavoritaLaggedCovariateTests(unittest.TestCase):
    def test_store_transactions_and_oil_are_shifted_one_day(self):
        dates=pd.date_range("2020-01-01",periods=3)
        frame=pd.DataFrame({"date":dates,"store_nbr":[1,1,1],"id":["1_10"]*3,
            "transactions":[100,200,300],"oil_price":[10.0,20.0,30.0]})
        result=add_lagged_covariates(frame).set_index("date")
        self.assertEqual(result.loc[dates[1],"transactions_lag_1"],100)
        self.assertEqual(result.loc[dates[2],"transactions_lag_1"],200)
        self.assertEqual(result.loc[dates[1],"oil_price_lag_1"],10.0)
        self.assertEqual(result.loc[dates[2],"oil_price_lag_1"],20.0)

    def test_same_day_changes_do_not_change_lagged_features(self):
        dates=pd.date_range("2020-01-01",periods=3)
        frame=pd.DataFrame({"date":dates,"store_nbr":[1]*3,"id":["1_10"]*3,
            "transactions":[100,200,300],"oil_price":[10.0,20.0,30.0]})
        changed=frame.copy(); changed.loc[2,"transactions"]=9999; changed.loc[2,"oil_price"]=9999
        a=add_lagged_covariates(frame).iloc[2]; b=add_lagged_covariates(changed).iloc[2]
        self.assertEqual(a.transactions_lag_1,b.transactions_lag_1)
        self.assertEqual(a.oil_price_lag_1,b.oil_price_lag_1)

    def test_future_exogenous_values_are_frozen_at_forecast_origin(self):
        origin=pd.Timestamp("2020-01-02")
        panel=pd.DataFrame({"date":[pd.Timestamp("2020-01-01"),origin],"id":["1_10","1_10"],
            "store_nbr":[1,1],"item_nbr":[10,10],"oil_price":[10.0,20.0],"oil_price_lag_1":[9.0,10.0]})
        testday=pd.DataFrame({"date":[pd.Timestamp("2020-01-03"),pd.Timestamp("2020-01-04")],
            "store_nbr":[1,1],"item_nbr":[10,10],"onpromotion":[False,False]})
        items=pd.DataFrame({"item_nbr":[10],"family":["FOOD"],"class":[1],"perishable":[0]})
        stores=pd.DataFrame({"store_nbr":[1],"state":["Pichincha"],"type":["A"],"cluster":[1]})
        oil=pd.DataFrame({"date":pd.date_range("2020-01-01",periods=4),"dcoilwtico":[10.0,20.0,999.0,888.0]})
        holidays=pd.DataFrame(columns=["date","locale","transferred"])
        transactions=pd.DataFrame({"date":pd.date_range("2020-01-01",periods=4),"store_nbr":[1]*4,
            "transactions":[100,200,9999,8888]})
        future=_future_exog(testday,panel,items,stores,oil,holidays,transactions)
        self.assertTrue((future.oil_price_lag_1==20.0).all())
        self.assertTrue((future.transactions_lag_1==200).all())


if __name__=="__main__": unittest.main()
