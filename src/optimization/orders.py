"""Scenario-based single-period inventory optimization using OR-Tools CP-SAT."""
from math import ceil
from statistics import NormalDist

SCENARIO_WEIGHTS = (25, 50, 25)  # percent weights for low, median, and high demand

def safety_stock_from_uncertainty(demand_std, service_level=0.90):
    """Normal-approximation safety stock for a lead-time/forecast horizon."""
    if not 0.5 < float(service_level) < 1:
        raise ValueError("service_level must be greater than 0.5 and less than 1")
    sigma=max(0.0,float(demand_std))
    z=NormalDist().inv_cdf(float(service_level))
    return max(0,int(ceil(z*sigma))),z

def _scenario_cost(order, on_hand, demands, holding_cents, stockout_cents):
    holding_weighted=0; stockout_weighted=0; ending_weighted=0; shortage_weighted=0
    for demand,weight in zip(demands,SCENARIO_WEIGHTS):
        ending=max(0,on_hand+order-demand)
        shortage=max(0,demand-on_hand-order)
        holding_weighted += weight*holding_cents*ending
        stockout_weighted += weight*stockout_cents*shortage
        ending_weighted += weight*ending
        shortage_weighted += weight*shortage
    # Weights sum to 100 and per-unit costs are in cents: divide by 10,000 for cost units.
    return {"estimated_holding_cost":holding_weighted/10000.0,
            "estimated_stockout_cost":stockout_weighted/10000.0,
            "total_objective_cost":(holding_weighted+stockout_weighted)/10000.0,
            "objective_cost_scaled":holding_weighted+stockout_weighted,
            "expected_ending_inventory":ending_weighted/100.0,
            "estimated_shortage":shortage_weighted/100.0}

def optimize_order(mean_demand, demand_std=0.0, on_hand=0, holding_cost=0.20,
                   stockout_cost=2.0, service_level=0.90, max_order=2000,
                   max_inventory=5000, baseline_order=None):
    """Choose an integer order minimizing expected holding and shortage costs.

    Demand uncertainty is approximated with a three-point scenario distribution at
    the configured service-level quantiles: low, median, and high, weighted 25/50/25.
    Costs are per unit over the replenishment horizon. `baseline_order` is an optional
    existing policy quantity evaluated against the same scenarios for comparison.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:
        raise RuntimeError("Install ortools to calculate optimized orders") from e
    mean=max(0.0,float(mean_demand)); sigma=max(0.0,float(demand_std)); inv=max(0,int(on_hand))
    hold=float(holding_cost); short=float(stockout_cost)
    if hold<0 or short<0: raise ValueError("holding_cost and stockout_cost must be non-negative")
    if int(max_order)<0 or int(max_inventory)<0: raise ValueError("inventory and order bounds must be non-negative")
    safety,z=safety_stock_from_uncertainty(sigma,service_level)
    mean_units=max(0,int(round(mean)))
    low=max(0,int(round(mean-z*sigma))); median=mean_units; high=max(median,int(ceil(mean+z*sigma)))
    demands=(low,median,high)
    max_q=min(int(max_order),max(0,int(max_inventory)-inv))
    holding_cents=int(round(hold*100)); stockout_cents=int(round(short*100))
    model=cp_model.CpModel()
    order=model.NewIntVar(0,max_q,"order_quantity")
    ending_vars=[]; shortage_vars=[]
    for i,demand in enumerate(demands):
        ending=model.NewIntVar(0,max(0,inv+max_q),f"ending_inventory_{i}")
        shortage=model.NewIntVar(0,demand,f"unmet_demand_{i}")
        model.Add(ending>=inv+order-demand)
        model.Add(shortage>=demand-inv-order)
        ending_vars.append(ending); shortage_vars.append(shortage)
    model.Minimize(sum(weight*(holding_cents*ending_vars[i]+stockout_cents*shortage_vars[i])
                       for i,weight in enumerate(SCENARIO_WEIGHTS)))
    solver=cp_model.CpSolver(); status=solver.Solve(model)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        raise RuntimeError(f"OR-Tools failed to find an inventory solution (status={solver.StatusName(status)})")
    q=solver.Value(order)
    baseline=max(0,min(max_q,int(round(baseline_order if baseline_order is not None else mean_units+safety-inv))))
    optimized_cost=_scenario_cost(q,inv,demands,holding_cents,stockout_cents)
    baseline_cost=_scenario_cost(baseline,inv,demands,holding_cents,stockout_cents)
    return {"solver_status":solver.StatusName(status),"mean_forecast_demand":mean_units,
      "demand_std_over_horizon":round(sigma,4),"service_level":float(service_level),"safety_stock":safety,
      "scenario_low_demand":low,"scenario_median_demand":median,"scenario_high_demand":high,
      "forecast_demand":mean_units+safety,"on_hand":inv,"inventory_position":inv,
      "max_order":int(max_order),"max_inventory":int(max_inventory),"recommended_order":q,
      "available_after_order":inv+q,"expected_ending_inventory":optimized_cost["expected_ending_inventory"],
      "estimated_shortage":optimized_cost["estimated_shortage"],
      "estimated_holding_cost":optimized_cost["estimated_holding_cost"],
      "estimated_stockout_cost":optimized_cost["estimated_stockout_cost"],
      "total_objective_cost":optimized_cost["total_objective_cost"],
      "objective_cost_scaled":optimized_cost["objective_cost_scaled"],
      "current_policy_order":baseline,"current_policy_holding_cost":baseline_cost["estimated_holding_cost"],
      "current_policy_stockout_cost":baseline_cost["estimated_stockout_cost"],
      "current_policy_objective_cost":baseline_cost["total_objective_cost"],
      "cost_scenario_weights":"25% low / 50% median / 25% high"}
