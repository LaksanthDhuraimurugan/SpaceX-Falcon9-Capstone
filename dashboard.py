"""Interactive Plotly Dash dashboard for Falcon 9 launch data."""
from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"outputs"/"wrangled_launch_data.csv"
if not DATA.exists():
    raise FileNotFoundError("Run python data_collection.py and python analysis_pipeline.py first.")
df=pd.read_csv(DATA)
df["PayloadMass"]=pd.to_numeric(df["PayloadMass"],errors="coerce")
df["Class"]=pd.to_numeric(df["Class"],errors="coerce")
sites=["All Sites"]+sorted(df["LaunchSite"].dropna().unique().tolist())
payload_min=float(df["PayloadMass"].min()) if df["PayloadMass"].notna().any() else 0
payload_max=float(df["PayloadMass"].max()) if df["PayloadMass"].notna().any() else 10000
if payload_min==payload_max: payload_max=payload_min+1
app=Dash(__name__)
app.layout=html.Div([
 html.H1("SpaceX Falcon 9 Launch Analysis",style={"textAlign":"center"}),
 html.Label("Select launch site"),
 dcc.Dropdown(id="site",options=[{"label":s,"value":s} for s in sites],value="All Sites",clearable=False),
 html.Label("Payload mass range (kg)"),
 dcc.RangeSlider(id="payload",min=payload_min,max=payload_max,step=max(1,(payload_max-payload_min)/100),
  value=[payload_min,payload_max],tooltip={"placement":"bottom","always_visible":False}),
 dcc.Graph(id="success-pie"),
 dcc.Graph(id="payload-outcome"),
 dcc.Graph(id="yearly-trend")
],style={"maxWidth":"1100px","margin":"auto","padding":"20px"})

@app.callback(Output("success-pie","figure"),Output("payload-outcome","figure"),Output("yearly-trend","figure"),
 Input("site","value"),Input("payload","value"))
def update(site,payload_range):
 d=df.copy()
 if site!="All Sites": d=d[d["LaunchSite"]==site]
 d=d[d["PayloadMass"].between(payload_range[0],payload_range[1]) | d["PayloadMass"].isna()]
 pie_data=d["Class"].map({1:"Successful landing",0:"Unsuccessful/other"}).value_counts().rename_axis("Outcome").reset_index(name="Count")
 pie=px.pie(pie_data,names="Outcome",values="Count",title="Landing Outcomes for Selected Site")
 scatter=px.scatter(d,x="PayloadMass",y="LaunchSite",color=d["Class"].map({1:"Success",0:"Unsuccessful/other"}),
  hover_data=["FlightNumber","BoosterVersion","Orbit"],title="Payload vs Launch Site and Landing Outcome")
 years=d.dropna(subset=["Year"]).groupby("Year",as_index=False)["Class"].mean()
 trend=px.line(years,x="Year",y="Class",markers=True,title="Yearly Landing Success Proportion")
 trend.update_yaxes(range=[0,1])
 return pie,scatter,trend

if __name__=="__main__":
 app.run(debug=True)
