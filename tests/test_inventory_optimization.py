import unittest

from src.optimization.orders import optimize_order, safety_stock_from_uncertainty


class InventoryOptimizationTests(unittest.TestCase):
    def test_zero_inventory_has_feasible_bounded_order_and_solver_runs(self):
        result=optimize_order(mean_demand=100,demand_std=20,on_hand=0,max_order=500,max_inventory=700)
        self.assertEqual(result["solver_status"],"OPTIMAL")
        self.assertGreater(result["recommended_order"],0)
        self.assertLessEqual(result["recommended_order"],500)
        self.assertLessEqual(result["available_after_order"],700)

    def test_sufficient_inventory_orders_zero_and_costs_holding(self):
        result=optimize_order(mean_demand=20,demand_std=0,on_hand=100,max_order=500)
        self.assertEqual(result["solver_status"],"OPTIMAL")
        self.assertEqual(result["recommended_order"],0)
        self.assertEqual(result["estimated_shortage"],0)
        self.assertGreater(result["estimated_holding_cost"],0)
        self.assertAlmostEqual(result["total_objective_cost"],result["estimated_holding_cost"]+result["estimated_stockout_cost"])

    def test_order_quantity_accounts_for_on_hand_stock(self):
        no_stock=optimize_order(mean_demand=100,demand_std=20,on_hand=0,max_order=500)
        stocked=optimize_order(mean_demand=100,demand_std=20,on_hand=30,max_order=500)
        self.assertEqual(stocked["recommended_order"],max(0,no_stock["recommended_order"]-30))

    def test_high_demand_respects_order_cap_and_reports_shortage_cost(self):
        result=optimize_order(mean_demand=1000,demand_std=100,on_hand=0,max_order=100,max_inventory=1000)
        self.assertEqual(result["recommended_order"],100)
        self.assertGreater(result["estimated_shortage"],0)
        self.assertGreater(result["estimated_stockout_cost"],0)
        self.assertGreater(result["total_objective_cost"],0)

    def test_inventory_capacity_binds_order_quantity(self):
        result=optimize_order(mean_demand=100,demand_std=10,on_hand=40,max_order=1000,max_inventory=50)
        self.assertLessEqual(result["recommended_order"],10)
        self.assertLessEqual(result["available_after_order"],50)

    def test_optimized_policy_cost_is_no_greater_than_current_policy(self):
        result=optimize_order(mean_demand=100,demand_std=20,on_hand=0,holding_cost=.2,stockout_cost=2,
                              max_order=500,baseline_order=200)
        self.assertLessEqual(result["total_objective_cost"],result["current_policy_objective_cost"])
        self.assertAlmostEqual(result["total_objective_cost"],
                               result["estimated_holding_cost"]+result["estimated_stockout_cost"])
        self.assertAlmostEqual(result["objective_cost_scaled"]/10000,result["total_objective_cost"])

    def test_solver_solution_matches_brute_force_minimum(self):
        result=optimize_order(mean_demand=100,demand_std=20,on_hand=15,holding_cost=.35,
                              stockout_cost=1.75,max_order=160,max_inventory=175)
        demands=[result["scenario_low_demand"],result["scenario_median_demand"],result["scenario_high_demand"]]
        weights=[.25,.50,.25]
        def cost(order):
            return sum(w*(.35*max(0,15+order-d)+1.75*max(0,d-15-order)) for d,w in zip(demands,weights))
        feasible=range(min(160,175-15)+1)
        brute_min=min(cost(q) for q in feasible)
        self.assertAlmostEqual(result["total_objective_cost"],brute_min)
        self.assertAlmostEqual(cost(result["recommended_order"]),brute_min)

    def test_cost_tradeoff_changes_optimal_quantity(self):
        stockout_sensitive=optimize_order(mean_demand=100,demand_std=20,holding_cost=.20,stockout_cost=2.0)
        holding_sensitive=optimize_order(mean_demand=100,demand_std=20,holding_cost=10.0,stockout_cost=1.0)
        self.assertLess(holding_sensitive["recommended_order"],stockout_sensitive["recommended_order"])
        self.assertGreater(holding_sensitive["estimated_stockout_cost"],0)

    def test_service_level_controls_uncertainty_buffer(self):
        ss90,_=safety_stock_from_uncertainty(25,.90)
        ss95,_=safety_stock_from_uncertainty(25,.95)
        self.assertGreater(ss95,ss90)

    def test_rejects_invalid_costs_and_service_levels(self):
        with self.assertRaises(ValueError): safety_stock_from_uncertainty(10,1.0)
        with self.assertRaises(ValueError): safety_stock_from_uncertainty(10,.5)
        with self.assertRaises(ValueError): optimize_order(10,holding_cost=-1)


if __name__=="__main__":
    unittest.main()
