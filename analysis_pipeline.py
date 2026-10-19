"""Wrangling, EDA, SQL, Folium maps, and classification for the Falcon 9 capstone."""
from pathlib import Path
import sqlite3, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

ROOT = Path(__file__).resolve().parent
DATA, OUT, MAPS = ROOT/"data", ROOT/"outputs", ROOT/"maps"
OUT.mkdir(exist_ok=True); MAPS.mkdir(exist_ok=True)

def load_data():
    candidates = [DATA/"launches_fallback.csv", DATA/"spacex_api_raw.csv", DATA/"dataset_part_2.csv"]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                print("Loaded", path, df.shape)
                return df
    raise FileNotFoundError("Run python data_collection.py first, or put a course launch CSV in data/.")

def first_existing(df, options):
    return next((x for x in options if x in df.columns), None)

def wrangle(df):
    # Standardize the common course dataset and API schema to a shared set of columns.
    aliases = {
        "FlightNumber": ["FlightNumber","flight_number"],
        "Date": ["Date","date_utc","launch_date_utc","date"],
        "BoosterVersion": ["BoosterVersion","rocket.name","rocket"],
        "PayloadMass": ["PayloadMass","payload_mass_kg","Payload Mass (kg)"],
        "Orbit": ["Orbit","orbit"],
        "LaunchSite": ["LaunchSite","launchpad.name","launch_site"],
        "Outcome": ["Outcome","landing_outcomes","cores.0.landing_success","landing_success"],
        "Latitude": ["Latitude","launchpad.latitude"],
        "Longitude": ["Longitude","launchpad.longitude"],
    }
    out = pd.DataFrame(index=df.index)
    for target, opts in aliases.items():
        col = first_existing(df, opts)
        out[target] = df[col] if col else np.nan

    # Nested API fields may need to be interpreted from expanded JSON columns.
    if out["Date"].isna().all() and "date_utc" in df: out["Date"] = df["date_utc"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True)
    out["Year"] = out["Date"].dt.year
    out["PayloadMass"] = pd.to_numeric(out["PayloadMass"], errors="coerce")
    out["FlightNumber"] = pd.to_numeric(out["FlightNumber"], errors="coerce")

    # Derive target from common course outcome strings or API landing_success.
    if out["Outcome"].notna().any():
        s = out["Outcome"].astype(str).str.lower()
        out["Class"] = s.str.contains(r"success|true|ground pad|drone ship", regex=True).astype(int)
        # For course outcomes, only explicit successful landing labels are positives.
        out.loc[s.str.contains(r"failure|no attempt|uncontrolled|false|nan"), "Class"] = 0
    else:
        col = first_existing(df, ["cores.0.landing_success", "landing_success"])
        if col:
            out["Class"] = df[col].map({True:1, False:0, "True":1, "False":0})
        else:
            out["Class"] = np.nan
    out["Class"] = pd.to_numeric(out["Class"], errors="coerce")
    # If course dataset already has the target variable, preserve it.
    target_col = first_existing(df, ["Class","class","Outcome_Success","LandingSuccess"])
    if target_col:
        out["Class"] = pd.to_numeric(df[target_col], errors="coerce").fillna(out["Class"])
    out["LaunchSite"] = out["LaunchSite"].fillna("Unknown").astype(str)
    out["Orbit"] = out["Orbit"].fillna("Unknown").astype(str)
    out["BoosterVersion"] = out["BoosterVersion"].fillna("Unknown").astype(str)
    out.to_csv(OUT/"wrangled_launch_data.csv", index=False)
    return out

def make_eda(df):
    sns.set_theme()
    valid = df.dropna(subset=["FlightNumber"])
    plots = [
        ("flight_vs_site.png", lambda: sns.scatterplot(data=valid, x="FlightNumber", y="LaunchSite", hue="Class")),
        ("payload_vs_site.png", lambda: sns.scatterplot(data=df, x="PayloadMass", y="LaunchSite", hue="Class")),
        ("success_by_orbit.png", lambda: sns.barplot(data=df, x="Orbit", y="Class", errorbar=None)),
        ("flight_vs_orbit.png", lambda: sns.scatterplot(data=df, x="FlightNumber", y="Orbit", hue="Class")),
        ("payload_vs_orbit.png", lambda: sns.scatterplot(data=df, x="PayloadMass", y="Orbit", hue="Class")),
    ]
    for name, draw in plots:
        try:
            plt.figure(figsize=(11,6)); draw(); plt.title(name.replace("_"," ").replace(".png","").title())
            plt.tight_layout(); plt.savefig(OUT/name, dpi=150); plt.close()
        except Exception as e:
            plt.close(); print("Skipped plot", name, e)
    yearly = df.dropna(subset=["Year"]).groupby("Year", as_index=False)["Class"].mean()
    if not yearly.empty:
        plt.figure(figsize=(9,5)); sns.lineplot(data=yearly, x="Year", y="Class", marker="o")
        plt.ylabel("Landing success proportion"); plt.title("Yearly Falcon 9 Landing Success Trend")
        plt.tight_layout(); plt.savefig(OUT/"yearly_success.png", dpi=150); plt.close()
        yearly.to_csv(OUT/"yearly_success.csv", index=False)

def make_sql(df):
    con = sqlite3.connect(OUT/"spacex.sqlite")
    df.to_sql("launches", con, if_exists="replace", index=False)
    queries = {
      "unique_launch_sites": "SELECT DISTINCT LaunchSite FROM launches ORDER BY LaunchSite",
      "cca_launch_sites": "SELECT * FROM launches WHERE LaunchSite LIKE 'CCA%' LIMIT 5",
      "total_payload_by_site": "SELECT LaunchSite, SUM(PayloadMass) AS total_payload_kg FROM launches GROUP BY LaunchSite ORDER BY total_payload_kg DESC",
      "average_payload_by_booster": "SELECT BoosterVersion, AVG(PayloadMass) AS avg_payload_kg FROM launches GROUP BY BoosterVersion ORDER BY avg_payload_kg DESC",
      "success_failure_counts": "SELECT Class, COUNT(*) AS count FROM launches GROUP BY Class",
      "max_payload_records": "SELECT BoosterVersion, PayloadMass, LaunchSite, Date FROM launches WHERE PayloadMass=(SELECT MAX(PayloadMass) FROM launches)",
      "yearly_counts": "SELECT Year, COUNT(*) AS launches, AVG(Class) AS success_rate FROM launches GROUP BY Year ORDER BY Year",
      "drone_ship_success_payload_4k_6k": "SELECT BoosterVersion, PayloadMass, Outcome FROM launches WHERE lower(Outcome) LIKE '%drone ship%' AND lower(Outcome) LIKE '%success%' AND PayloadMass > 4000 AND PayloadMass < 6000",
      "2015_records": "SELECT Date, LaunchSite, BoosterVersion, Outcome FROM launches WHERE Year=2015 AND lower(Outcome) LIKE '%drone ship%' AND lower(Outcome) LIKE '%fail%'",
    }
    with open(OUT/"sql_results.txt","w",encoding="utf-8") as f:
        for name, query in queries.items():
            f.write(f"\\n## {name}\\n{query}\\n")
            try: f.write(pd.read_sql_query(query,con).to_string(index=False)+"\\n")
            except Exception as e: f.write(f"Query note: {e}\\n")
    con.close()

def make_maps(df):
    # Known launch-pad coordinates, useful if source lacks geospatial fields.
    coords = {
      "CCAFS SLC 40": (28.5619,-80.5774), "Cape Canaveral": (28.5619,-80.5774),
      "KSC LC 39A": (28.6082,-80.6041), "Kennedy Space Center": (28.6082,-80.6041),
      "VAFB SLC 4E": (34.632,-120.611), "Vandenberg": (34.632,-120.611),
      "Kwajalein Atoll": (9.0477,167.743), "Unknown": (0,0)
    }
    m = folium.Map(location=[25, -35], zoom_start=2)
    for site, group in df.groupby("LaunchSite"):
        latcol, loncol = group["Latitude"].dropna(), group["Longitude"].dropna()
        if len(latcol) and len(loncol): lat, lon = float(latcol.iloc[0]), float(loncol.iloc[0])
        else:
            found = next((v for k,v in coords.items() if k.lower() in site.lower() or site.lower() in k.lower()), None)
            if not found or site == "Unknown": continue
            lat, lon = found
        folium.Marker([lat,lon], tooltip=site, popup=f"{site}<br>Launches: {len(group)}").add_to(m)
    m.save(MAPS/"launch_sites.html")
    # outcome map using site coordinates and green/red markers
    mo = folium.Map(location=[25,-35], zoom_start=2)
    for _, row in df.iterrows():
        site = str(row["LaunchSite"])
        found = next((v for k,v in coords.items() if k.lower() in site.lower() or site.lower() in k.lower()), None)
        lat = row["Latitude"] if pd.notna(row["Latitude"]) else (found[0] if found else None)
        lon = row["Longitude"] if pd.notna(row["Longitude"]) else (found[1] if found else None)
        if pd.notna(lat) and pd.notna(lon):
            color = "green" if row["Class"] == 1 else "red"
            folium.CircleMarker([float(lat),float(lon)], radius=5, color=color, fill=True,
              popup=f"{site} | success={row['Class']}").add_to(mo)
    mo.save(MAPS/"launch_outcomes.html")

def model_data(df):
    data = df.dropna(subset=["Class"]).copy()
    data = data[data["Class"].isin([0,1])]
    features = ["PayloadMass","FlightNumber","Year","Orbit","LaunchSite","BoosterVersion"]
    X = data[features].copy()
    y = data["Class"].astype(int)
    # Require enough samples and both classes.
    if len(data) < 12 or y.nunique() < 2:
        print("Not enough labelled rows/classes for reliable ML. Ensure course dataset_part_2.csv is available.")
        return
    numeric = ["PayloadMass","FlightNumber","Year"]
    categorical = ["Orbit","LaunchSite","BoosterVersion"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=42,stratify=stratify)
    models = {
      "Logistic Regression": LogisticRegression(max_iter=2000),
      "SVM": SVC(),
      "Decision Tree": DecisionTreeClassifier(random_state=42),
      "KNN": KNeighborsClassifier(n_neighbors=min(5,max(1,len(X_train)//4)))
    }
    rows=[]; fitted={}
    for name, model in models.items():
        pipe=Pipeline([("preprocess",pre),("model",model)])
        pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
        acc=accuracy_score(y_test,pred); rows.append({"Model":name,"Accuracy":acc})
        fitted[name]=(pipe, pred)
        print(f"\\n{name}: accuracy={acc:.3f}")
        print(classification_report(y_test,pred,zero_division=0))
    scores=pd.DataFrame(rows).sort_values("Accuracy",ascending=False)
    scores.to_csv(OUT/"model_accuracy.csv",index=False)
    plt.figure(figsize=(8,5)); sns.barplot(data=scores,x="Model",y="Accuracy")
    plt.ylim(0,1); plt.title("Classification Model Accuracy"); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(OUT/"model_accuracy.png",dpi=150); plt.close()
    best=scores.iloc[0]["Model"]; pipe,pred=fitted[best]
    joblib.dump(pipe,OUT/"best_landing_model.joblib")
    ConfusionMatrixDisplay.from_predictions(y_test,pred)
    plt.title(f"Confusion Matrix - {best}"); plt.tight_layout()
    plt.savefig(OUT/"best_confusion_matrix.png",dpi=150); plt.close()
    (OUT/"best_model.txt").write_text(f"Best model by held-out test accuracy: {best}\\nAccuracy: {scores.iloc[0]['Accuracy']:.4f}\\n\\nThis score is based on a single seeded split; report it as an evaluation result, not a guarantee of future performance.",encoding="utf-8")

def main():
    raw=load_data()
    df=wrangle(raw)
    print("Wrangled data preview:\\n",df.head())
    make_eda(df); make_sql(df); make_maps(df); model_data(df)
    print(f"Done. Results in {OUT}; maps in {MAPS}.")
if __name__=="__main__": main()
