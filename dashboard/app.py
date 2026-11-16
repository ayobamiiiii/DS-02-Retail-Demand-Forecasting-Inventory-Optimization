from pathlib import Path
import json
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from src.evaluation.metrics import score

OUT=Path("outputs"); app=Dash(__name__, title="Retail Demand & Inventory")
def load_csv(name):
    p=OUT/name
    if not p.exists(): return pd.DataFrame()
    frame=pd.read_csv(p)
    if "date" in frame: frame["date"]=pd.to_datetime(frame.date)
    return frame
forecast=load_csv("forecasts/forecast.csv")
val=load_csv("forecasts/model_comparison_validation.csv")
if val.empty: val=load_csv("forecasts/validation.csv")
orders=load_csv("recommendations/orders.csv")
history=load_csv("forecasts/sales_history.csv")
try:
    model_metrics_path=OUT/"metrics/model_comparison.json"
    if not model_metrics_path.exists(): model_metrics_path=OUT/"metrics/model_metrics.json"
    metrics=json.loads(model_metrics_path.read_text())
except Exception: metrics={}
stores=sorted(forecast.store_nbr.dropna().unique()) if not forecast.empty and "store_nbr" in forecast else []
items=sorted(forecast.item_nbr.dropna().unique()) if not forecast.empty and "item_nbr" in forecast else []

def verify_heldout_metrics(saved, validation):
    """Recompute metrics from saved held-out predictions and pre-holdout history."""
    panel_path=Path("data/processed/favorita_panel.csv")
    if validation.empty or not panel_path.exists() or "date" not in validation:
        return False,"Held-out metric verification unavailable: validation predictions or training panel are missing."
    panel=pd.read_csv(panel_path,usecols=["date","id","unit_sales"],parse_dates=["date"])
    cutoff=validation.date.min()
    train_by_id={k:g.sort_values("date").unit_sales.to_numpy() for k,g in panel[panel.date<cutoff].groupby("id",observed=True)}
    actual_check=validation[["date","id","actual"]].merge(panel,on=["date","id"],how="left",validate="one_to_one")
    if actual_check.unit_sales.isna().any() or not (actual_check.actual.to_numpy()==actual_check.unit_sales.to_numpy()).all():
        return False,"Held-out metric verification failed: saved actuals do not match the processed training panel."
    comparisons={"naive_weekly":"prediction_naive_weekly","lightgbm":"prediction_lightgbm",
                 "prophet":"prediction_prophet","darts_regression":"prediction_darts_regression","tft":"prediction_tft"}
    comparisons={k:v for k,v in comparisons.items() if k in saved and v in validation}
    if "naive_weekly" in saved and "naive_weekly" not in comparisons and "lag_7" in validation:
        comparisons["naive_weekly"]="lag_7"
    if "lightgbm" in saved and "lightgbm" not in comparisons and "prediction_lgbm" in validation:
        comparisons["lightgbm"]="prediction_lgbm"
    for model_name,pred_col in comparisons.items():
        if model_name not in saved or pred_col not in validation:
            return False,f"Held-out metric verification unavailable for {model_name}."
        part=validation[["id","actual",pred_col]].rename(columns={"actual":"sales",pred_col:"prediction"})
        observed=score(part,"prediction",train_by_id)
        for metric in ("wape","rmse","rmsse_mean"):
            expected=saved[model_name].get(metric)
            if expected is None or not abs(float(expected)-float(observed[metric]))<=1e-6:
                return False,f"Held-out metric verification failed for {model_name} ({metric})."
    if validation.date.min()<=panel[panel.date<cutoff].date.max():
        return False,"Held-out metric verification failed: validation dates overlap the training period."
    return True,"Verified by recomputing WAPE, RMSSE, and RMSE from the saved held-out predictions and pre-validation training history."

metrics_verified,metrics_verification_message=verify_heldout_metrics(metrics,val)

def model_label(name):
    labels={"naive_weekly":"Naive Weekly","lightgbm":"LightGBM","prophet":"Prophet",
            "darts_regression":"Darts Regression","tft":"Temporal Fusion Transformer"}
    return labels.get(name,name.replace("_"," ").title())

def model_metric_rows():
    rows=[]
    for name,result in metrics.items():
        if isinstance(result,dict) and all(k in result for k in ("wape","rmsse_mean","rmse")):
            rows.append({"key":name,"model":model_label(name),"WAPE":float(result["wape"]),
                         "RMSSE":float(result["rmsse_mean"]),"RMSE":float(result["rmse"])})
    return pd.DataFrame(rows)

def performance_panel():
    table=model_metric_rows()
    if table.empty:
        return html.P("Model metrics are unavailable. Run the forecasting pipeline to generate outputs/metrics/model_metrics.json.")
    fields=[("WAPE","WAPE","lower is better"),("RMSSE","RMSSE","lower is better"),("RMSE","RMSE","lower is better")]
    best={metric:table.loc[table[metric].idxmin()] for metric,_,_ in fields}
    kpis=[]
    for metric,title,hint in fields:
        value=float(best[metric][metric]); formatted=f"{value:.2%}" if metric=="WAPE" else f"{value:.3f}"
        kpis.append(html.Div([html.Div(f"Best {title}",style={"fontSize":"13px","color":"#596579"}),
          html.Div(formatted,style={"fontSize":"28px","fontWeight":"700","color":"#17324d"}),
          html.Div(f"{best[metric]['model']} · {hint}",style={"fontSize":"13px","color":"#287a54"})],
          style={"background":"#f4f7fa","border":"1px solid #dbe3ec","borderRadius":"10px","padding":"16px","flex":"1","minWidth":"170px"}))
    charts=[]
    for metric,title,_ in fields:
        chart=px.bar(table,x="model",y=metric,color="model",text=metric,
          color_discrete_map={row.model:("#21936b" if row.model==best[metric]["model"] else "#9aa8b8") for _,row in table.iterrows()},
          title=f"{title} by model",hover_data={"model":True,metric:":.4f"})
        chart.update_traces(texttemplate="%{y:.2%}" if metric=="WAPE" else "%{y:.3f}",textposition="outside")
        chart.update_layout(showlegend=False,margin={"l":20,"r":15,"t":55,"b":20},yaxis_title=metric,
          yaxis_tickformat=".0%" if metric=="WAPE" else ".3f",xaxis_title=None,plot_bgcolor="white",paper_bgcolor="white")
        charts.append(dcc.Graph(figure=chart,config={"displayModeBar":True,"displaylogo":False},style={"flex":"1","minWidth":"280px"}))
    sorted_rows=table.sort_values("WAPE")
    interpretation=(f"{sorted_rows.iloc[0].model} has the lowest WAPE at {sorted_rows.iloc[0].WAPE:.2%}. "
      f"{best['RMSSE']['model']} has the lowest RMSSE ({best['RMSSE']['RMSSE']:.3f}), and "
      f"{best['RMSE']['model']} has the lowest RMSE ({best['RMSE']['RMSE']:.3f}).")
    if len(table)>1:
        baseline=table[table.key=="naive_weekly"]
        winner=table.loc[table.WAPE.idxmin()]
        if not baseline.empty and float(baseline.iloc[0].WAPE)>0:
            reduction=(float(baseline.iloc[0].WAPE)-float(winner.WAPE))/float(baseline.iloc[0].WAPE)
            if winner.key!="naive_weekly": interpretation+=f" That is a {reduction:.1%} WAPE reduction versus Naive Weekly."
    if not val.empty and "date" in val:
        period=f"{val.date.min():%b %d, %Y} – {val.date.max():%b %d, %Y}"
        n_series=int(val.id.nunique()) if "id" in val else int(max(r.get("n_series",0) for r in metrics.values() if isinstance(r,dict)))
        n_rows=len(val)
    else: period="Unavailable"; n_series=0; n_rows=0
    provenance_color="#287a54" if metrics_verified else "#a76518"
    design=metrics.get("model_comparison",{})
    selection_period=design.get("selection_period",{}); calibration_period=design.get("calibration_period",{})
    protocol_note=None
    if selection_period and calibration_period:
        protocol_note=html.P(f"Model selection: {selection_period.get('start')} to {selection_period.get('end')} · "
          f"interval calibration: {calibration_period.get('start')} to {calibration_period.get('end')} · "
          "evaluation: untouched final period",style={"fontSize":"13px","color":"#596579"})
    execution=design.get("execution",{})
    execution_note=None
    if execution:
        omitted=[f"{model_label(name)}: {detail.get('reason',detail.get('status','not evaluated'))}"
                 for name,detail in execution.items() if detail.get("status") not in ("executed",)]
        if omitted:
            execution_note=html.P("Model execution status · " + " · ".join(omitted),
                                  style={"fontSize":"13px","color":"#8a5a13"})
    return html.Div([
      html.Div(kpis,style={"display":"flex","gap":"14px","flexWrap":"wrap","marginBottom":"18px"}),
      html.P(interpretation,style={"background":"#f8fafc","borderLeft":"4px solid #4b84b6","padding":"12px 16px","margin":"8px 0 16px"}),
      html.Div([html.Strong(f"Evaluated {n_series:,} series · {n_rows:,} held-out observations · {period}"),
        html.Div(metrics_verification_message,style={"color":provenance_color,"fontSize":"13px","marginTop":"5px"})],
        style={"marginBottom":"14px"}),
      protocol_note,execution_note,
      html.Div(charts,style={"display":"flex","gap":"12px","flexWrap":"wrap"})])

def inventory_panel(frame):
    if frame.empty:
        return html.P("No inventory recommendations match the current store and item filters.")
    numeric=("recommended_order","estimated_holding_cost","estimated_stockout_cost","total_objective_cost",
             "current_policy_order","current_policy_objective_cost")
    for col in numeric:
        if col not in frame: frame[col]=0.0
    totals={c:float(frame[c].sum()) for c in numeric}
    cards=[]
    for title,value,fmt in (("Optimized order units",totals["recommended_order"],"units"),
      ("Expected holding cost",totals["estimated_holding_cost"],"cost"),
      ("Expected stockout cost",totals["estimated_stockout_cost"],"cost"),
      ("Optimized total cost",totals["total_objective_cost"],"cost")):
        shown=f"{value:,.0f}" if fmt=="units" else f"{value:,.2f} cost units"
        cards.append(html.Div([html.Div(title,style={"fontSize":"13px","color":"#596579"}),
          html.Div(shown,style={"fontSize":"25px","fontWeight":"700","color":"#17324d"})],
          style={"background":"#f4f7fa","border":"1px solid #dbe3ec","borderRadius":"10px","padding":"15px","flex":"1","minWidth":"180px"}))
    plot=frame.copy()
    plot["item-store"]=plot.item_nbr.astype(str)+" · S"+plot.store_nbr.astype(str)
    plot=plot.nlargest(25,"forecast_demand")
    inv_fig=px.bar(plot,x="item-store",y=[c for c in ("forecast_demand","on_hand","recommended_order","available_after_order") if c in plot],
      barmode="group",title="Demand, available inventory, and order · top 25 by forecast",
      labels={"value":"Units","variable":"Quantity","item-store":"Item · store"},
      color_discrete_map={"forecast_demand":"#4b84b6","on_hand":"#9aa8b8","recommended_order":"#21936b","available_after_order":"#64b7a0"})
    inv_fig.update_layout(plot_bgcolor="white",paper_bgcolor="white",legend_title_text="",margin={"t":55,"b":35})
    policy=pd.DataFrame([{"Policy":"Current order-up-to","Order units":totals["current_policy_order"],"Expected cost":totals["current_policy_objective_cost"]},
                         {"Policy":"OR-Tools optimized","Order units":totals["recommended_order"],"Expected cost":totals["total_objective_cost"]}])
    cost_fig=px.bar(policy,x="Policy",y="Expected cost",color="Policy",text="Expected cost",title="Expected scenario cost by policy",
      color_discrete_map={"Current order-up-to":"#9aa8b8","OR-Tools optimized":"#21936b"})
    cost_fig.update_traces(texttemplate="%{y:,.2f}",textposition="outside")
    cost_fig.update_layout(plot_bgcolor="white",paper_bgcolor="white",showlegend=False,margin={"t":55,"b":25},yaxis_title="Expected cost units")
    brief=(f"The current order-up-to rule recommends {totals['current_policy_order']:,.0f} units with estimated scenario cost "
      f"{totals['current_policy_objective_cost']:,.2f}; the OR-Tools policy recommends {totals['recommended_order']:,.0f} units "
      f"with expected holding plus stockout cost {totals['total_objective_cost']:,.2f}. Costs use configured assumptions and a "
      f"{float(frame.service_level.iloc[0]):.0%} service level.")
    cols=[("item_nbr","Item"),("store_nbr","Store"),("mean_forecast_demand","Mean demand"),("safety_stock","Safety stock"),
      ("on_hand","On hand"),("current_policy_order","Current order"),("recommended_order","Optimized order"),
      ("expected_ending_inventory","Expected ending"),("estimated_shortage","Expected shortage"),
      ("estimated_holding_cost","Holding cost"),("estimated_stockout_cost","Stockout cost"),("total_objective_cost","Total cost")]
    header=html.Tr([html.Th(label,style={"textAlign":"left","padding":"7px","borderBottom":"1px solid #dbe3ec"}) for key,label in cols])
    body=[]
    for row in frame.sort_values("forecast_demand",ascending=False).head(50).to_dict("records"):
        cells=[]
        for key,_ in cols:
            value=row.get(key,"")
            shown=f"{value:,.2f}" if key in {"expected_ending_inventory","estimated_shortage","estimated_holding_cost","estimated_stockout_cost","total_objective_cost"} else f"{value:,.0f}" if isinstance(value,(int,float)) else value
            cells.append(html.Td(shown,style={"padding":"7px","borderBottom":"1px solid #edf1f5"}))
        body.append(html.Tr(cells))
    table=html.Div(html.Table([header]+body,style={"borderCollapse":"collapse","width":"100%","fontSize":"13px"}),style={"overflowX":"auto"})
    return html.Div([html.Div(cards,style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"15px"}),
      html.P(brief,style={"background":"#f8fafc","borderLeft":"4px solid #4b84b6","padding":"12px 16px"}),
      dcc.Graph(figure=inv_fig,config={"displayModeBar":True,"displaylogo":False}),
      dcc.Graph(figure=cost_fig,config={"displayModeBar":True,"displaylogo":False}),html.H3("Policy detail by item and store"),table])
app.layout=html.Div([
 html.H1("Retail Demand Forecasting & Inventory Planning"),
 html.P("Favorita grocery sales · bottom-up coherent rollups · empirical residual intervals"),
 html.Div([html.Label("Store"),dcc.Dropdown(stores,stores,multi=True,id="stores"),html.Label("Item"),dcc.Dropdown(items,items,multi=True,id="items"),html.Label("Date range"),
     dcc.DatePickerRange(id="dates",start_date=forecast.date.min() if not forecast.empty else None,end_date=forecast.date.max() if not forecast.empty else None)],style={"maxWidth":"900px"}),
 dcc.Tabs(id="tabs",value="Executive Overview",children=[dcc.Tab(label=x,value=x) for x in ["Executive Overview","Sales Trends","Demand Forecasts","Model Performance","Inventory Optimization","Store and Item Analysis"]]),
 html.Div(id="content",style={"padding":"20px"})],style={"fontFamily":"Arial","maxWidth":"1200px","margin":"auto"})
@app.callback(Output("content","children"),Input("stores","value"),Input("items","value"),Input("dates","start_date"),Input("dates","end_date"),Input("tabs","value"))
def update(stores,items,start,end,tab):
    if forecast.empty: return html.Div([html.H3("No forecast outputs yet"),html.P("Add Favorita CSV files to data/raw/favorita and run python3 -m src.pipeline.")])
    df=forecast.copy()
    if stores and "store_nbr" in df: df=df[df.store_nbr.isin(stores)]
    if items and "item_nbr" in df: df=df[df.item_nbr.isin(items)]
    if start: df=df[df.date>=pd.to_datetime(start)]
    if end: df=df[df.date<=pd.to_datetime(end)]
    daily=df.groupby("date",as_index=False)[["forecast","lower","upper"]].sum()
    fig=px.line(daily,x="date",y=["forecast","lower","upper"],title="Forecast with 90% interval")
    filtered_orders=orders.copy()
    if stores and "store_nbr" in filtered_orders: filtered_orders=filtered_orders[filtered_orders.store_nbr.isin(stores)]
    if items and "item_nbr" in filtered_orders: filtered_orders=filtered_orders[filtered_orders.item_nbr.isin(items)]
    cards=[html.Div(f"Forecast units: {df.forecast.sum():,.0f}"),html.Div(f"Stores: {df.store_nbr.nunique()}" if "store_nbr" in df else "Stores: unavailable"),html.Div(f"Recommended order units: {filtered_orders.recommended_order.sum():,.0f}" if not filtered_orders.empty else "Orders: unavailable")]
    children=[html.H2(tab)]
    if tab=="Executive Overview": children += [html.Div(cards,style={"display":"flex","gap":"30px","fontWeight":"bold"}),dcc.Graph(figure=fig)]
    elif tab=="Sales Trends":
        h=history.copy()
        if stores and "store_nbr" in h: h=h[h.store_nbr.isin(stores)]
        if items and "item_nbr" in h: h=h[h.item_nbr.isin(items)]
        if start: h=h[h.date>=pd.to_datetime(start)]
        if end: h=h[h.date<=pd.to_datetime(end)]
        children.append(dcc.Graph(figure=px.line(h,x="date",y="sales",title="Historical sales") if not h.empty else fig))
    elif tab=="Demand Forecasts": children.append(dcc.Graph(figure=fig))
    elif tab=="Model Performance": children.append(performance_panel())
    elif tab=="Inventory Optimization": children.append(inventory_panel(filtered_orders))
    elif tab=="Store and Item Analysis":
        children.append(html.Pre(df.groupby(["store_nbr","item_nbr"],as_index=False).forecast.sum().nlargest(30,"forecast").to_string(index=False)))
    return html.Div(children)
if __name__=="__main__": app.run(debug=False)
