from pathlib import Path
import json
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
from dash.dash_table.Format import Format, Scheme, Group
import plotly.express as px
import plotly.graph_objects as go
from src.evaluation.metrics import score

OUT=Path("outputs"); app=Dash(__name__, title="Retail Demand & Inventory",suppress_callback_exceptions=True)
def load_csv(name):
    p=OUT/name
    if not p.exists(): return pd.DataFrame()
    frame=pd.read_csv(p)
    if "date" in frame: frame["date"]=pd.to_datetime(frame.date)
    return frame
forecast=load_csv("forecasts/forecast.csv")
val=load_csv("forecasts/model_comparison_validation.csv")
if val.empty: val=load_csv("forecasts/validation.csv")
primary_validation=load_csv("forecasts/validation.csv")
if primary_validation.empty: primary_validation=val.copy()
orders=load_csv("recommendations/orders.csv")
history=load_csv("forecasts/sales_history.csv")
try:
    model_metrics_path=OUT/"metrics/model_comparison.json"
    if not model_metrics_path.exists(): model_metrics_path=OUT/"metrics/model_metrics.json"
    metrics=json.loads(model_metrics_path.read_text())
except Exception: metrics={}
store_values=set()
item_values=set()
for source in (history,forecast):
    if not source.empty and "store_nbr" in source: store_values.update(source.store_nbr.dropna().unique().tolist())
    if not source.empty and "item_nbr" in source: item_values.update(source.item_nbr.dropna().unique().tolist())
stores=sorted(store_values); items=sorted(item_values)
available_dates=[]
for source in (history,primary_validation,forecast):
    if not source.empty and "date" in source: available_dates.extend([source.date.min(),source.date.max()])
date_min=min(available_dates) if available_dates else None
date_max=max(available_dates) if available_dates else None
default_date_start=(forecast.date.min()-pd.DateOffset(years=1)) if not forecast.empty else date_min

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

def filter_frame(frame,stores_selected,items_selected,start,end):
    """Apply the same global scope and date controls to an output table."""
    result=frame.copy()
    if result.empty: return result
    if stores_selected and "store_nbr" in result: result=result[result.store_nbr.isin(stores_selected)]
    if items_selected and "item_nbr" in result: result=result[result.item_nbr.isin(items_selected)]
    if start and "date" in result: result=result[result.date>=pd.to_datetime(start)]
    if end and "date" in result: result=result[result.date<=pd.to_datetime(end)]
    return result

def metric_card(title,value):
    return html.Div([html.Div(title,style={"fontSize":"13px","color":"#596579"}),
      html.Div(value,style={"fontSize":"25px","fontWeight":"700","color":"#17324d"})],
      style={"background":"#f4f7fa","border":"1px solid #dbe3ec","borderRadius":"10px","padding":"15px","flex":"1","minWidth":"180px"})

def store_item_summary_values(rows):
    """Summarize the table's currently filtered rows, not the unfiltered source."""
    frame=pd.DataFrame(rows or [])
    if frame.empty:
        return "0.00","0","0"
    return (f"{pd.to_numeric(frame.forecast_units,errors='coerce').fillna(0).sum():,.2f}",
      f"{frame.store_nbr.nunique():,}",f"{frame.item_nbr.nunique():,}")

def store_item_analysis_view(frame):
    """Render globally filtered forecasts as a searchable, sortable data table."""
    if frame.empty:
        aggregated=pd.DataFrame(columns=["store_nbr","item_nbr","forecast_units"])
    else:
        aggregated=(frame.groupby(["store_nbr","item_nbr"],as_index=False,observed=True).forecast.sum()
          .rename(columns={"forecast":"forecast_units"})
          .sort_values("forecast_units",ascending=False).reset_index(drop=True))
    total_units,store_count,item_count=store_item_summary_values(aggregated.to_dict("records"))
    table=dash_table.DataTable(
      id="store-item-analysis-table",
      columns=[
        {"name":"Store Number","id":"store_nbr","type":"numeric"},
        {"name":"Item Number","id":"item_nbr","type":"numeric"},
        {"name":"Forecast Units","id":"forecast_units","type":"numeric",
         "format":Format(precision=2,scheme=Scheme.fixed,group=Group.yes)},
      ],
      data=aggregated.to_dict("records"),
      sort_action="native",sort_mode="multi",sort_by=[{"column_id":"forecast_units","direction":"desc"}],
      filter_action="native",filter_options={"case_sensitive":False},
      page_action="native",page_size=20,
      style_table={"overflowX":"auto","border":"1px solid #dbe3ec","borderRadius":"8px"},
      style_header={"backgroundColor":"#f4f7fa","color":"#17324d","fontWeight":"700",
        "borderBottom":"2px solid #dbe3ec","padding":"12px 14px"},
      style_cell={"padding":"11px 14px","fontFamily":"Arial","fontSize":"14px",
        "color":"#243b53","borderBottom":"1px solid #edf1f5","height":"auto",
        "minWidth":"130px","width":"33%","maxWidth":"none"},
      style_cell_conditional=[
        {"if":{"column_id":"store_nbr"},"textAlign":"left"},
        {"if":{"column_id":"item_nbr"},"textAlign":"left"},
        {"if":{"column_id":"forecast_units"},"textAlign":"right"},
      ],
      style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#fbfcfe"}],
    )
    return html.Div([
      html.Div([html.Div([html.Div("Total forecast units",style={"fontSize":"13px","color":"#596579"}),
          html.Div(total_units,id="store-item-total-units",style={"fontSize":"25px","fontWeight":"700","color":"#17324d"})],
          style={"background":"#f4f7fa","border":"1px solid #dbe3ec","borderRadius":"10px","padding":"15px","flex":"1","minWidth":"180px"}),
        html.Div([html.Div("Stores displayed",style={"fontSize":"13px","color":"#596579"}),
          html.Div(store_count,id="store-item-stores-count",style={"fontSize":"25px","fontWeight":"700","color":"#17324d"})],
          style={"background":"#f4f7fa","border":"1px solid #dbe3ec","borderRadius":"10px","padding":"15px","flex":"1","minWidth":"180px"}),
        html.Div([html.Div("Items displayed",style={"fontSize":"13px","color":"#596579"}),
          html.Div(item_count,id="store-item-items-count",style={"fontSize":"25px","fontWeight":"700","color":"#17324d"})],
          style={"background":"#f4f7fa","border":"1px solid #dbe3ec","borderRadius":"10px","padding":"15px","flex":"1","minWidth":"180px"})],
        style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"16px"}),
      html.P("Forecast totals reflect the current store, item, and date selections. Use the column filters to search the results and sort any column.",
        style={"background":"#f8fafc","borderLeft":"4px solid #4b84b6","padding":"10px 14px"}),
      table])

@app.callback(
  Output("store-item-total-units","children"),
  Output("store-item-stores-count","children"),
  Output("store-item-items-count","children"),
  Input("store-item-analysis-table","derived_virtual_data"),
)
def update_store_item_summary(rows):
    """Keep KPI cards in sync with native table search/filter results."""
    return store_item_summary_values(rows)

def sales_trends_view(frame):
    """Build charts and KPIs from observed historical Favorita sales only."""
    if frame.empty:
        return html.P("No historical sales match the current store, item, and date filters.")
    daily=frame.groupby("date",as_index=False).sales.sum().sort_values("date")
    total=float(daily.sales.sum())
    average=float(daily.sales.mean())
    peak=daily.loc[daily.sales.idxmax()]
    cards=[metric_card("Actual sales units",f"{total:,.0f}"),
      metric_card("Average actual units / day",f"{average:,.1f}"),
      metric_card("Peak sales day",f"{peak.date:%b %d, %Y} · {peak.sales:,.0f} units"),
      metric_card("Stores in view",f"{frame.store_nbr.nunique():,}")]
    trend=px.line(daily,x="date",y="sales",title="Daily historical sales · actuals only",
      labels={"date":"Date","sales":"Actual units"})
    trend.update_traces(line={"color":"#4b84b6","width":2},hovertemplate="%{x|%b %d, %Y}<br>Actual sales: %{y:,.1f}<extra></extra>")
    trend.update_layout(plot_bgcolor="white",paper_bgcolor="white",margin={"t":55,"b":25},showlegend=False)

    weekdays=frame.assign(weekday=frame.date.dt.dayofweek).groupby("weekday",as_index=False).sales.mean()
    weekday_names=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekdays=weekdays.set_index("weekday").reindex(range(7),fill_value=0).rename_axis("weekday").reset_index()
    weekdays["weekday_name"]=[weekday_names[int(day)] for day in weekdays.weekday]
    weekly=px.bar(weekdays,x="weekday_name",y="sales",title="Average actual sales by weekday",
      labels={"weekday_name":"Day of week","sales":"Average actual units"},color_discrete_sequence=["#21936b"])
    weekly.update_layout(plot_bgcolor="white",paper_bgcolor="white",showlegend=False,margin={"t":55,"b":25})

    by_store=frame.groupby(["date","store_nbr"],as_index=False).sales.sum()
    store_fig=px.line(by_store,x="date",y="sales",color="store_nbr",title="Historical sales by selected store",
      labels={"date":"Date","sales":"Actual units","store_nbr":"Store"})
    store_fig.update_layout(plot_bgcolor="white",paper_bgcolor="white",margin={"t":55,"b":25},legend_title_text="Store")
    options={"displayModeBar":True,"displaylogo":False}
    return html.Div([
      html.Div(cards,style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"16px"}),
      html.P("All values and charts on this tab come from observed Favorita sales; forecasts and prediction intervals are excluded.",
        style={"background":"#f8fafc","borderLeft":"4px solid #4b84b6","padding":"10px 14px"}),
      dcc.Graph(figure=trend,config=options),
      html.Div([dcc.Graph(figure=weekly,config=options,style={"flex":"1","minWidth":"320px"}),
        dcc.Graph(figure=store_fig,config=options,style={"flex":"2","minWidth":"420px"})],
        style={"display":"flex","gap":"14px","flexWrap":"wrap"})])

def demand_forecasts_view(history_frame,validation_frame,future_frame):
    """Show actual history, saved rolling validation predictions, and future forecasts."""
    validation_start=primary_validation.date.min() if not primary_validation.empty else pd.Timestamp.max
    history_plot=history_frame[history_frame.date < validation_start]
    fig=go.Figure()
    if not history_plot.empty:
        daily=history_plot.groupby("date",as_index=False).sales.sum().sort_values("date")
        fig.add_trace(go.Scatter(x=daily.date,y=daily.sales,name="Historical actual sales",mode="lines",
          line={"color":"#52606d","width":1.7},hovertemplate="%{x|%b %d, %Y}<br>Actual: %{y:,.1f}<extra></extra>"))
    if not validation_frame.empty:
        actual=validation_frame.groupby("date",as_index=False).actual.sum().sort_values("date")
        predicted=validation_frame.groupby("date",as_index=False).forecast.sum().sort_values("date")
        fig.add_trace(go.Scatter(x=actual.date,y=actual.actual,name="Validation actual sales",mode="lines",
          line={"color":"#17324d","width":2.4},hovertemplate="%{x|%b %d, %Y}<br>Actual: %{y:,.1f}<extra></extra>"))
        model_name="Selected-model validation prediction"
        fig.add_trace(go.Scatter(x=predicted.date,y=predicted.forecast,name=model_name,mode="lines",
          line={"color":"#e28a2c","width":2.2,"dash":"dash"},hovertemplate="%{x|%b %d, %Y}<br>Predicted: %{y:,.1f}<extra></extra>"))
    if not future_frame.empty:
        future_daily=future_frame.groupby("date",as_index=False)[[c for c in ("forecast","lower","upper") if c in future_frame]].sum().sort_values("date")
        if "lower" in future_daily and "upper" in future_daily:
            fig.add_trace(go.Scatter(x=future_daily.date,y=future_daily.lower,name="90% interval lower",mode="lines",
              line={"width":0},showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=future_daily.date,y=future_daily.upper,name="90% prediction interval",mode="lines",
              line={"width":0},fill="tonexty",fillcolor="rgba(75,132,182,0.18)",
              hovertemplate="%{x|%b %d, %Y}<br>Interval upper: %{y:,.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=future_daily.date,y=future_daily.forecast,name="Future model forecast",mode="lines",
          line={"color":"#4b84b6","width":2.5},hovertemplate="%{x|%b %d, %Y}<br>Forecast: %{y:,.1f}<extra></extra>"))
    fig.update_layout(title="Actual sales, held-out predictions, and future forecast",xaxis_title="Date",yaxis_title="Units",
      plot_bgcolor="white",paper_bgcolor="white",legend_title_text="Series",hovermode="x unified",margin={"t":60,"b":30})
    cards=[]
    if not validation_frame.empty:
        cards.extend([metric_card("Validation actual units",f"{validation_frame.actual.sum():,.0f}"),
          metric_card("Validation predicted units",f"{validation_frame.forecast.sum():,.0f}")])
    if not future_frame.empty:
        cards.append(metric_card("Future forecast units",f"{future_frame.forecast.sum():,.0f}"))
    note="Validation values compare saved predictions with known actuals; future dates show model forecasts and calibrated intervals, with no actuals available."
    if fig.data:
        return html.Div([html.Div(cards,style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"14px"}),
          html.P(note,style={"background":"#f8fafc","borderLeft":"4px solid #4b84b6","padding":"10px 14px"}),
          dcc.Graph(figure=fig,config={"displayModeBar":True,"displaylogo":False})])
    return html.P("No actual sales or forecast rows match the current store, item, and date filters.")
app.layout=html.Div([
 html.H1("Retail Demand Forecasting & Inventory Planning"),
 html.P("Favorita grocery sales · bottom-up coherent rollups · empirical residual intervals"),
 html.Div([html.Label("Store"),dcc.Dropdown(stores,stores,multi=True,id="stores"),html.Label("Item"),dcc.Dropdown(items,items,multi=True,id="items"),html.Label("Date range"),
     dcc.DatePickerRange(id="dates",start_date=default_date_start,end_date=date_max,
       min_date_allowed=date_min,max_date_allowed=date_max)],style={"maxWidth":"900px"}),
 dcc.Tabs(id="tabs",value="Executive Overview",children=[dcc.Tab(label=x,value=x) for x in ["Executive Overview","Sales Trends","Demand Forecasts","Model Performance","Inventory Optimization","Store and Item Analysis"]]),
 html.Div(id="content",style={"padding":"20px"})],style={"fontFamily":"Arial","maxWidth":"1200px","margin":"auto"})
@app.callback(Output("content","children"),Input("stores","value"),Input("items","value"),Input("dates","start_date"),Input("dates","end_date"),Input("tabs","value"))
def update(stores,items,start,end,tab):
    if forecast.empty: return html.Div([html.H3("No forecast outputs yet"),html.P("Add Favorita CSV files to data/raw/favorita and run python3 -m src.pipeline.")])
    df=filter_frame(forecast,stores,items,start,end)
    historical=filter_frame(history,stores,items,start,end)
    validation_frame=filter_frame(primary_validation,stores,items,start,end)
    filtered_orders=orders.copy()
    if stores and "store_nbr" in filtered_orders: filtered_orders=filtered_orders[filtered_orders.store_nbr.isin(stores)]
    if items and "item_nbr" in filtered_orders: filtered_orders=filtered_orders[filtered_orders.item_nbr.isin(items)]
    cards=[metric_card("Forecast units",f"{df.forecast.sum():,.0f}"),
      metric_card("Stores in view",f"{df.store_nbr.nunique():,}" if "store_nbr" in df else "Unavailable"),
      metric_card("Recommended order units",f"{filtered_orders.recommended_order.sum():,.0f}" if not filtered_orders.empty else "Unavailable"),
      metric_card("Modeled inventory cost",f"{filtered_orders.total_objective_cost.sum():,.2f}" if not filtered_orders.empty else "Unavailable")]
    children=[html.H2(tab)]
    if tab=="Executive Overview":
        children += [html.Div(cards,style={"display":"flex","gap":"12px","flexWrap":"wrap"}),
          html.P("Use Sales Trends for observed daily and weekday patterns. Demand Forecasts separates historical actuals, held-out predictions, and future forecasts.",
            style={"background":"#f8fafc","borderLeft":"4px solid #4b84b6","padding":"12px 16px","marginTop":"16px"})]
    elif tab=="Sales Trends": children.append(sales_trends_view(historical))
    elif tab=="Demand Forecasts": children.append(demand_forecasts_view(historical,validation_frame,df))
    elif tab=="Model Performance": children.append(performance_panel())
    elif tab=="Inventory Optimization": children.append(inventory_panel(filtered_orders))
    elif tab=="Store and Item Analysis": children.append(store_item_analysis_view(df))
    return html.Div(children)
if __name__=="__main__": app.run(debug=False)
