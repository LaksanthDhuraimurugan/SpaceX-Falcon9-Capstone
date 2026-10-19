
from pathlib import Path
import warnings

import pandas as pd
import folium
import requests

from folium.plugins import MarkerCluster, Fullscreen, MiniMap
from shapely.geometry import Point, LineString
from shapely.ops import transform
from pyproj import Transformer

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MAP_DIR = ROOT / "maps"

MAP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# BASEMAP CONFIGURATION
# ---------------------------------------------------------

# Esri World Light Gray Canvas basemap.
# This avoids the blocked OpenStreetMap tile configuration
# and the CARTO API-key error seen previously.

BASEMAP_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)

BASEMAP_ATTRIBUTION = (
    "Tiles © Esri — Sources: Esri, HERE, Garmin, "
    "© OpenStreetMap contributors, and the GIS User Community"
)


def create_base_map(location, zoom_start):
    """Create a map using the shared Esri basemap."""
    return folium.Map(
        location=location,
        zoom_start=zoom_start,
        tiles=BASEMAP_URL,
        attr=BASEMAP_ATTRIBUTION,
        control_scale=True
    )


# ---------------------------------------------------------
# KNOWN SPACEX LAUNCH SITE COORDINATES
# ---------------------------------------------------------

KNOWN_SITES = {
    "CCAFS SLC 40": (28.5619, -80.5774),
    "CCAFS LC-40": (28.5619, -80.5774),
    "KSC LC 39A": (28.6082, -80.6041),
    "VAFB SLC 4E": (34.6320, -120.6110),
    "Kwajalein Atoll": (9.0477, 167.7430),
}


# ---------------------------------------------------------
# FIND A COLUMN WITHOUT CASE SENSITIVITY
# ---------------------------------------------------------

def find_column(df, names):
    columns = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in names:
        if name.lower() in columns:
            return columns[name.lower()]

    return None


# ---------------------------------------------------------
# IDENTIFY A KNOWN LAUNCH SITE
# ---------------------------------------------------------

def identify_site(site):
    text = str(site).lower()

    if any(x in text for x in [
        "slc-40", "slc 40", "ccafs", "cape canaveral"
    ]):
        return KNOWN_SITES["CCAFS SLC 40"]

    if any(x in text for x in ["39a", "kennedy", "ksc"]):
        return KNOWN_SITES["KSC LC 39A"]

    if any(x in text for x in [
        "vafb", "vandenberg", "slc-4e", "slc 4e"
    ]):
        return KNOWN_SITES["VAFB SLC 4E"]

    if "kwajalein" in text:
        return KNOWN_SITES["Kwajalein Atoll"]

    return None


# ---------------------------------------------------------
# LOAD THE EXISTING DATASET
# ---------------------------------------------------------

def load_data():
    candidates = [
        OUTPUT_DIR / "wrangled_launch_data.csv",
        DATA_DIR / "dataset_part_2.csv",
        DATA_DIR / "launches_fallback.csv",
        DATA_DIR / "spacex_api_raw.csv",
        ROOT / "spacex_launch_data.csv",
        ROOT / "dataset_part_2.csv",
    ]

    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)

                if not df.empty:
                    print(f"Dataset loaded: {path}")
                    print(f"Rows: {len(df)}")
                    print(f"Columns: {list(df.columns)}")
                    return df

            except Exception as exc:
                warnings.warn(
                    f"Could not read {path}: {exc}"
                )

    raise FileNotFoundError(
        "No usable CSV found. Check the paths in load_data(). "
        "Run your data_collection.py first."
    )


# ---------------------------------------------------------
# PREPARE AND NORMALIZE DATA
# ---------------------------------------------------------

def prepare_data(raw):
    site_col = find_column(
        raw,
        ["LaunchSite", "launch_site", "launchpad.name"]
    )

    class_col = find_column(
        raw,
        ["Class", "class", "LandingSuccess", "Outcome_Success"]
    )

    outcome_col = find_column(
        raw,
        ["Outcome", "landing_outcomes", "landing_success"]
    )

    date_col = find_column(
        raw, ["Date", "date", "date_utc"]
    )

    flight_col = find_column(
        raw, ["FlightNumber", "flight_number"]
    )

    lat_col = find_column(
        raw, ["Latitude", "latitude", "launchpad.latitude"]
    )

    lon_col = find_column(
        raw, ["Longitude", "longitude", "launchpad.longitude"]
    )

    if site_col is None:
        raise ValueError(
            "No launch-site column found. Available columns: "
            + str(list(raw.columns))
        )

    df = pd.DataFrame(index=raw.index)

    df["LaunchSite"] = (
        raw[site_col].fillna("Unknown").astype(str)
    )

    df["Class"] = (
        pd.to_numeric(raw[class_col], errors="coerce")
        if class_col is not None
        else float("nan")
    )

    df["Outcome"] = (
        raw[outcome_col].fillna("Unknown").astype(str)
        if outcome_col is not None
        else "Not supplied"
    )

    df["Date"] = (
        raw[date_col].fillna("").astype(str)
        if date_col is not None
        else ""
    )

    df["FlightNumber"] = (
        raw[flight_col]
        if flight_col is not None
        else ""
    )

    df["Latitude"] = (
        pd.to_numeric(raw[lat_col], errors="coerce")
        if lat_col is not None
        else float("nan")
    )

    df["Longitude"] = (
        pd.to_numeric(raw[lon_col], errors="coerce")
        if lon_col is not None
        else float("nan")
    )

    # Fill missing coordinates using known launch sites.
    for idx, row in df.iterrows():
        coords = identify_site(row["LaunchSite"])

        if coords:
            if pd.isna(row["Latitude"]):
                df.at[idx, "Latitude"] = coords[0]

            if pd.isna(row["Longitude"]):
                df.at[idx, "Longitude"] = coords[1]

    # Reject invalid coordinates.
    invalid = (
        df["Latitude"].isna()
        | df["Longitude"].isna()
        | ~df["Latitude"].between(-90, 90)
        | ~df["Longitude"].between(-180, 180)
        | (
            (df["Latitude"] == 0)
            & (df["Longitude"] == 0)
        )
    )

    df.loc[
        invalid, ["Latitude", "Longitude"]
    ] = float("nan")

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# MAP CONTROLS
# ---------------------------------------------------------

def add_controls(map_object):
    Fullscreen(position="topright").add_to(map_object)
    MiniMap(toggle_display=True).add_to(map_object)


# =========================================================
# MAP 1: GLOBAL LAUNCH-SITE LOCATIONS
# =========================================================

def create_map_1(df):
    m = create_base_map(
        location=[25, -35],
        zoom_start=2
    )

    valid = df.dropna(
        subset=["Latitude", "Longitude"]
    )

    for site, group in valid.groupby("LaunchSite"):
        lat = float(group["Latitude"].median())
        lon = float(group["Longitude"].median())

        class_counts = group["Class"].value_counts()

        class_1 = int(class_counts.get(1, 0))
        class_0 = int(class_counts.get(0, 0))

        popup_html = f"""
        <b>Launch site:</b> {site}<br>
        <b>Records:</b> {len(group)}<br>
        <b>Class 1 records:</b> {class_1}<br>
        <b>Class 0 records:</b> {class_0}<br>
        <small>Counts use the selected project CSV.</small>
        """

        folium.Marker(
            location=[lat, lon],
            tooltip=str(site),
            popup=folium.Popup(
                popup_html,
                max_width=300
            ),
            icon=folium.Icon(
                color="blue",
                icon="info-sign"
            )
        ).add_to(m)

    add_controls(m)
    folium.LayerControl().add_to(m)

    m.save(
        MAP_DIR / "folium_map_1_launch_sites.html"
    )

    print("Created Map 1: global launch-site locations")


# =========================================================
# MAP 2: LANDING OUTCOMES
# =========================================================

def create_map_2(df):
    m = create_base_map(
        location=[25, -35],
        zoom_start=2
    )

    cluster = MarkerCluster(
        name="Launch records"
    ).add_to(m)

    valid = df.dropna(
        subset=["Latitude", "Longitude"]
    )

    for _, row in valid.iterrows():
        value = row["Class"]

        if value == 1:
            color = "green"
            label = "Class 1"

        elif value == 0:
            color = "red"
            label = "Class 0"

        else:
            color = "gray"
            label = "Unknown class"

        popup_html = f"""
        <b>Launch site:</b> {row['LaunchSite']}<br>
        <b>Flight:</b> {row['FlightNumber']}<br>
        <b>Date:</b> {row['Date']}<br>
        <b>Outcome:</b> {row['Outcome']}<br>
        <b>Class:</b> {label}
        """

        folium.CircleMarker(
            location=[
                row["Latitude"],
                row["Longitude"]
            ],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=f"{row['LaunchSite']} - {label}",
            popup=folium.Popup(
                popup_html,
                max_width=320
            )
        ).add_to(cluster)

    # Landing class legend.
    legend = """
    <div style="
        position:fixed;
        bottom:25px;
        left:25px;
        z-index:9999;
        background:white;
        padding:12px;
        border:1px solid grey;
        border-radius:5px;">
      <b>Landing class</b><br>
      <span style="color:green;">●</span> Class 1<br>
      <span style="color:red;">●</span> Class 0<br>
      <span style="color:gray;">●</span> Unknown
    </div>
    """

    m.get_root().html.add_child(
        folium.Element(legend)
    )

    add_controls(m)

    m.save(
        MAP_DIR / "folium_map_2_landing_outcomes.html"
    )

    print("Created Map 2: landing outcome distribution")


# =========================================================
# MAP 3: SELECTED SITE AND NEARBY GEOGRAPHIC FEATURES
# =========================================================

def get_osm_features():
    """
    Retrieve nearby coastline, highway and railway geometries
    from the Overpass API.

    Returns None if the service is unavailable. The map still
    works without the retrieved geographic features.
    """

    query = """
    [out:json][timeout:25];
    (
      way["natural"="coastline"](28.45,-80.75,28.75,-80.45);
      way["highway"](28.45,-80.75,28.75,-80.45);
      way["railway"](28.45,-80.75,28.75,-80.45);
    );
    out geom;
    """

    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=35,
            headers={
                "User-Agent": "SpaceX-Falcon9-Capstone/1.0"
            }
        )

        response.raise_for_status()

        elements = response.json().get(
            "elements", []
        )

        features = {
            "coastline": [],
            "highway": [],
            "railway": []
        }

        for element in elements:
            if "geometry" not in element:
                continue

            tags = element.get("tags", {})

            coords = [
                (p["lon"], p["lat"])
                for p in element["geometry"]
            ]

            if len(coords) < 2:
                continue

            line = LineString(coords)

            if tags.get("natural") == "coastline":
                features["coastline"].append(line)

            elif tags.get("railway"):
                features["railway"].append(line)

            elif tags.get("highway"):
                features["highway"].append(line)

        print(
            "Retrieved nearby OpenStreetMap features."
        )

        return features

    except Exception as exc:
        print(
            "OpenStreetMap feature retrieval unavailable:",
            exc
        )

        return None


# ---------------------------------------------------------
# DISTANCE TO THE NEAREST GEOGRAPHIC FEATURE
# ---------------------------------------------------------

def distance_to_nearest_feature_km(
    lat, lon, geometries
):
    """
    Calculate approximate straight-line distance to the
    nearest returned feature using a local metric projection.
    """

    if not geometries:
        return None

    project = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32617",
        always_xy=True
    ).transform

    point = transform(
        project,
        Point(lon, lat)
    )

    projected = [
        transform(project, geometry)
        for geometry in geometries
    ]

    distances = [
        point.distance(geometry)
        for geometry in projected
    ]

    return round(min(distances) / 1000, 2)


# =========================================================
# CREATE MAP 3
# =========================================================

def create_map_3(df):
    # Select the Cape Canaveral / Kennedy Space Center area.
    selected = df[
        df["LaunchSite"].str.contains(
            r"CCAFS|Cape Canaveral|SLC.?40|Kennedy|KSC|39A",
            case=False,
            na=False,
            regex=True
        )
    ]

    # Default coordinates for SLC-40.
    lat, lon = 28.5619, -80.5774
    site_name = "Cape Canaveral / SLC-40"

    if not selected.empty:
        row = selected.dropna(
            subset=["Latitude", "Longitude"]
        )

        if not row.empty:
            lat = float(
                row.iloc[0]["Latitude"]
            )

            lon = float(
                row.iloc[0]["Longitude"]
            )

            site_name = str(
                row.iloc[0]["LaunchSite"]
            )

    m = create_base_map(
        location=[lat, lon],
        zoom_start=12
    )

    # Selected launch-site marker.
    folium.Marker(
        [lat, lon],
        tooltip=site_name,
        popup=(
            f"<b>Selected launch site:</b> {site_name}"
        ),
        icon=folium.Icon(
            color="blue",
            icon="info-sign"
        )
    ).add_to(m)

    # Reference circles indicate map scale.
    # They are not measured feature distances.
    folium.Circle(
        [lat, lon],
        radius=5000,
        color="blue",
        fill=True,
        fill_opacity=0.04,
        tooltip="5 km reference radius"
    ).add_to(m)

    folium.Circle(
        [lat, lon],
        radius=10000,
        color="orange",
        fill=True,
        fill_opacity=0.03,
        tooltip="10 km reference radius"
    ).add_to(m)

    # Retrieve nearby geographic features.
    features = get_osm_features()

    summary = [
        f"<b>Selected site:</b> {site_name}",
        "<b>Distance type:</b> straight-line distance"
    ]

    if features is not None:
        styles = {
            "coastline": ("#168aad", "Coastline"),
            "highway": ("#d97706", "Highway"),
            "railway": ("#7e22ce", "Railway")
        }

        for key, (color, label) in styles.items():
            geometries = features[key]

            # Draw the retrieved features.
            for geometry in geometries:
                coords = [
                    (y, x)
                    for x, y in geometry.coords
                ]

                folium.PolyLine(
                    coords,
                    color=color,
                    weight=2,
                    tooltip=label
                ).add_to(m)

            # Calculate distance only if features exist.
            distance = distance_to_nearest_feature_km(
                lat,
                lon,
                geometries
            )

            if distance is None:
                summary.append(
                    f"{label}: not found in retrieved data"
                )

            else:
                summary.append(
                    f"{label}: {distance} km"
                )

        html = "<br>".join(summary)

        folium.Marker(
            [lat + 0.012, lon],
            icon=folium.DivIcon(
                html=(
                    "<div style='background:white;padding:8px;"
                    "border:1px solid #777;width:260px;"
                    "font-size:12px'>"
                    + html
                    + "</div>"
                )
            )
        ).add_to(m)

    else:
        summary.extend([
            "Nearby roads/railways/coastline could not be retrieved.",
            "Distance values were not fabricated.",
            "The circles show map scale only."
        ])

        folium.Marker(
            [lat + 0.012, lon],
            icon=folium.DivIcon(
                html=(
                    "<div style='background:white;padding:8px;"
                    "border:1px solid #777;width:250px;"
                    "font-size:12px'>"
                    + "<br>".join(summary)
                    + "</div>"
                )
            )
        ).add_to(m)

    add_controls(m)

    m.save(
        MAP_DIR / "folium_map_3_site_proximity.html"
    )

    print(
        "Created Map 3: selected site and geographic context"
    )


# =========================================================
# RUN ALL THREE MAPS
# =========================================================

def main():
    raw = load_data()
    df = prepare_data(raw)

    print("\nLaunch sites found:")
    print(
        df["LaunchSite"].value_counts().to_string()
    )

    print(
        "\nRows with coordinates:",
        int(
            df[
                ["Latitude", "Longitude"]
            ].notna().all(axis=1).sum()
        )
    )

    create_map_1(df)
    create_map_2(df)
    create_map_3(df)

    print("\nMaps saved in:", MAP_DIR)


if __name__ == "__main__":
    main()
