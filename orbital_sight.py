"""
OrbitalSight - Final Hackathon Submission
-----------------------------------------
AI-powered orbital debris visualization, clustering & risk prediction.

Features:
- Load TLEs (from CelesTrak or local file fallback)
- Orbit propagation with sgp4
- DBSCAN clustering of debris snapshots
- KDTree-based close-approach detection
- Risk heatmaps (lat/lon density)
- Interactive Dash dashboard (tabs for Overview, 3D, Heatmap, Close Approaches)
"""

import math
import logging
from datetime import datetime, timedelta, UTC
from typing import List, Tuple
from scipy.signal import find_peaks


import requests
import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

import plotly.graph_objects as go
from dash import Dash, dcc, html

# ----------------------------
# CONFIG
# ----------------------------
CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
FALLBACK_FILE = "debris_sample.tle"

DEFAULT_TIMESTEP_SEC = 60
PREDICTION_HOURS = 24
PROPAGATION_DURATION_HOURS = 6
DBSCAN_EPS_KM = 300.0
DBSCAN_MIN_SAMPLES = 3
CLOSE_APPROACH_RADIUS_KM = 100.0
MAX_DEBRIS_TO_USE = 800   # keep light for hackathon demo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------
# TLE LOADER
# ----------------------------
def load_tles(url: str, fallback_file: str) -> List[Tuple[str, str, str]]:
    """Load TLEs from URL, fallback to local file if URL fails."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
        logging.info(f"TLEs downloaded successfully from {url}")
    except Exception as e:
        logging.warning(f"Failed to download TLEs: {e}. Using fallback {fallback_file}")
        with open(fallback_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    tles = []
    i = 0
    while i < len(lines) - 2:
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            tles.append((name, l1, l2))
            i += 3
        else:
            i += 1
    return tles


# ----------------------------
# PROPAGATION
# ----------------------------
def propagate_single_tle(tle, epoch, duration_hours, dt_sec=DEFAULT_TIMESTEP_SEC):
    """Propagate a single TLE using sgp4."""
    name, l1, l2 = tle
    sat = Satrec.twoline2rv(l1, l2)
    n_steps = int((duration_hours * 3600) / dt_sec) + 1
    records = []

    for i in range(n_steps):
        t = epoch + timedelta(seconds=i * dt_sec)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e, r, _ = sat.sgp4(jd, fr)
        if e != 0:  # skip errors
            continue
        x, y, z = r
        rnorm = math.sqrt(x * x + y * y + z * z)
        lat = math.degrees(math.asin(z / rnorm))
        lon = math.degrees(math.atan2(y, x))
        records.append((name, t, x, y, z, lat, lon, rnorm))

    return pd.DataFrame(records, columns=["name", "time", "x", "y", "z", "lat", "lon", "r_km"])


def propagate_all(tles, epoch, duration_hours, max_objects=MAX_DEBRIS_TO_USE):
    """Propagate multiple TLEs in parallel."""
    tles_small = tles[:max_objects]
    results = Parallel(n_jobs=4)(
        delayed(propagate_single_tle)(tle, epoch, duration_hours) for tle in tles_small
    )
    return pd.concat(results, ignore_index=True)


def snapshot(df_positions, epoch, tol_seconds=30):
    """Extract snapshot at given epoch."""
    df = df_positions.copy()
    df["dt"] = (df["time"] - epoch).abs().dt.total_seconds()
    df_snap = df[df["dt"] <= tol_seconds].sort_values(["name", "dt"]).groupby("name").first().reset_index()
    return df_snap[["name", "time", "x", "y", "z", "lat", "lon", "r_km"]]


# ----------------------------
# CLUSTERING & CLOSE APPROACH
# ----------------------------
def cluster_snapshot(df_snap):
    """Cluster snapshot using DBSCAN."""
    xyz = df_snap[["x", "y", "z"]].values
    db = DBSCAN(eps=DBSCAN_EPS_KM, min_samples=DBSCAN_MIN_SAMPLES).fit(xyz)
    df_snap["cluster"] = db.labels_
    return df_snap


def detect_close_approaches(debris_positions, sat_positions, threshold_km=CLOSE_APPROACH_RADIUS_KM):
    """
    Detect close approaches between satellite and debris objects.
    Fixes:
    - Prevents satellite matching itself (dist=0.0 bug).
    - Ignores unrealistic overlaps (<0.1 km).
    - Uses smoother logistic risk formula (0–100).
    """
    debris = debris_positions.copy()
    sat = sat_positions.copy()
    debris["time_bucket"] = debris["time"].dt.floor(f"{DEFAULT_TIMESTEP_SEC}s")
    sat["time_bucket"] = sat["time"].dt.floor(f"{DEFAULT_TIMESTEP_SEC}s")

    results = []
    grouped = {
        k: {
            "positions": np.vstack([g["x"].values, g["y"].values, g["z"].values]).T,
            "names": g["name"].values,
            "df": g,
        }
        for k, g in debris.groupby("time_bucket")
    }

    for _, row in sat.iterrows():
        tb = row["time_bucket"]
        if tb not in grouped:
            continue
        tree = cKDTree(grouped[tb]["positions"])
        sat_pos = np.array([row["x"], row["y"], row["z"]])
        idxs = tree.query_ball_point(sat_pos, r=threshold_km)

        for i in idxs:
            debris_name = grouped[tb]["names"][i]

            # Skip if debris is actually the satellite itself
            if debris_name == row.get("sat_name", "SAT"):
                continue

            dist = np.linalg.norm(sat_pos - grouped[tb]["positions"][i])

            # Skip unrealistic overlaps
            if dist < 1.0:
                continue

            # Estimate relative velocity
            if i < len(grouped[tb]["positions"]) - 1:
                vel = np.linalg.norm(
                    grouped[tb]["positions"][i] - grouped[tb]["positions"][i - 1]
                )
            else:
                vel = 0.1  # fallback

            # Smooth logistic-style risk formula
            raw_score = (vel / (dist + 1e-3)) * 10
            risk = 100 / (1 + np.exp(-0.1 * (raw_score - 50)))

            results.append({
                "sat_time": row["time"],
                "sat_name": row.get("sat_name", "SAT"),
                "debris_name": debris_name,
                "distance_km": round(float(dist), 3),
                "relative_velocity": round(float(vel), 3),
                "risk_score": round(float(risk), 2),
            })

    return pd.DataFrame(results)





# ----------------------------
# HEATMAP
# ----------------------------
def build_globe(df_snap, title="Orbital Debris Density (Globe View)"):
    """3D globe-style visualization of debris density."""
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=df_snap["lon"],
        lat=df_snap["lat"],
        mode="markers",
        marker=dict(
            size=2,
            color=df_snap["cluster"],
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="Cluster ID")
        ),
        text=df_snap["name"]
    ))

    fig.update_layout(
        title=title,
        geo=dict(
            projection_type="orthographic",   # globe-style
            showland=True,
            landcolor="rgb(10,10,30)",
            showocean=True,
            oceancolor="rgb(0,0,50)",
            showlakes=True,
            lakecolor="rgb(0,0,70)",
            bgcolor="black"
        ),
        paper_bgcolor="black",
        font_color="white"
    )
    return fig


# ----------------------------
# VISUALIZATION
# ----------------------------
def build_3d_scatter(df):
    """Build 3D scatter plot of objects, colored by cluster."""
    fig = go.Figure()
    if "cluster" in df.columns:
        for label in sorted(df["cluster"].unique()):
            dfl = df[df["cluster"] == label]
            name = f"Cluster {label}" if label != -1 else "Noise"
            fig.add_trace(
                go.Scatter3d(
                    x=dfl["x"], y=dfl["y"], z=dfl["z"],
                    mode="markers", marker=dict(size=2),
                    name=name
                )
            )
    else:
        fig.add_trace(go.Scatter3d(x=df["x"], y=df["y"], z=df["z"], mode="markers"))
    fig.update_layout(scene=dict(aspectmode="data"), paper_bgcolor="black", font_color="white")
    return fig
def build_orbit_animation(df, title="Orbital Debris Time-Lapse"):
    """Animate debris orbits over time."""
    fig = go.Figure()

    # Group by name (each object gets a trace)
    for name, group in df.groupby("name"):
        fig.add_trace(go.Scatter3d(
            x=group["x"], y=group["y"], z=group["z"],
            mode="markers",
            marker=dict(size=2),
            name=name,
            text=group["time"].dt.strftime("%H:%M:%S"),
            visible=False
        ))

    # Animation frames
    frames = []
    timesteps = sorted(df["time"].unique())
    for t in timesteps:
        frame_data = []
        for name, group in df.groupby("name"):
            sub = group[group["time"] == t]
            frame_data.append(go.Scatter3d(
                x=sub["x"], y=sub["y"], z=sub["z"],
                mode="markers",
                marker=dict(size=2),
                name=name
            ))
        frames.append(go.Frame(data=frame_data, name=str(t)))

    fig.frames = frames

    fig.update_layout(
        scene=dict(aspectmode="data"),
        paper_bgcolor="black",
        font_color="white",
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}],
                 "label": "▶ Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                 "label": "⏸ Pause", "method": "animate"}
            ],
            "direction": "left", "pad": {"r": 10, "t": 87},
            "showactive": False, "type": "buttons", "x": 0.1, "y": 0, "xanchor": "right", "yanchor": "top"
        }]
    )

    return fig
def build_orbit_animation(df, title="Orbital Debris Time-Lapse (2D Globe)"):
    """Animate debris orbits on a 2D globe projection."""
    timesteps = sorted(df["time"].unique())

    frames = []
    for t in timesteps:
        frame_data = go.Scattergeo(
            lon=df[df["time"] == t]["lon"],
            lat=df[df["time"] == t]["lat"],
            mode="markers",
            marker=dict(size=2, color="cyan"),
            text=df[df["time"] == t]["name"]
        )
        frames.append(go.Frame(data=[frame_data], name=str(t)))

    fig = go.Figure(
        data=[frames[0].data[0]],
        frames=frames
    )

    fig.update_layout(
        title=title,
        geo=dict(
            projection_type="orthographic",
            showland=True, landcolor="rgb(10,10,30)",
            showocean=True, oceancolor="rgb(0,0,50)",
            bgcolor="black"
        ),
        paper_bgcolor="black", font_color="white",
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 200, "redraw": True},
                                 "fromcurrent": True, "mode": "immediate"}],
                 "label": "▶ Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate"}],
                 "label": "⏸ Pause", "method": "animate"}
            ],
            "direction": "left",
            "x": 0.1, "y": -0.1,
            "showactive": False, "type": "buttons"
        }],
        sliders=[{
            "steps": [
                {"args": [[str(t)], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate"}],
                 "label": str(t.time()), "method": "animate"}
                for t in timesteps
            ],
            "x": 0.1, "y": -0.15,
            "len": 0.9,
            "currentvalue": {"prefix": "Time: "}
        }]
    )
    return fig
def build_risk_heatmap(close_df: pd.DataFrame, df_positions: pd.DataFrame,
                       lat_bins:int=72, lon_bins:int=144, top_n_hotspots:int=8):
    """
    Build a weighted risk heatmap (flat lat/lon) with contour overlay and hotspots.
    - close_df: must include 'risk_score' and 'sat_time' and 'debris_name'
    - df_positions: full propagated positions (df_all) used to map coords
    """
    if close_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Close Approaches → No Risk Zones")
        return fig

    # Map close entries to lat/lon
    mapped = map_close_to_coords(close_df, df_positions)
    # drop rows without coords
    mapped = mapped.dropna(subset=['lat','lon'])
    if mapped.empty:
        fig = go.Figure()
        fig.update_layout(title="No matched coordinates for close approaches")
        return fig

    # Use risk_score as weight if available, else weight=1
    weights = mapped['risk_score'].values if 'risk_score' in mapped.columns else np.ones(len(mapped))

    # 2D histogram (lat, lon)
    heat, lat_edges, lon_edges = np.histogram2d(mapped['lat'].values, mapped['lon'].values,
                                                bins=[lat_bins, lon_bins], range=[[-90,90],[-180,180]],
                                                weights=weights)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    # Make Plotly figure: heatmap + contour
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=heat.T, x=lon_centers, y=lat_centers,
        colorscale='YlOrRd', colorbar=dict(title='Weighted risk')
    ))

    # Add contours to highlight zones (transparent fill)
    max_z = float(np.nanmax(heat))
    if max_z > 0:
        # choose contour levels (3 levels)
        levels = np.linspace(max_z*0.2, max_z, 4)
        fig.add_trace(go.Contour(
            z=heat.T, x=lon_centers, y=lat_centers,
            contours=dict(start=levels[0], end=levels[-1], size=(levels[-1]-levels[0])/3),
            showscale=False, opacity=0.45, colorscale='Reds', hoverinfo='skip'
        ))

    # Identify hotspots (local max bins)
    # flatten and pick top N indices
    flat = heat.T.ravel()
    if len(flat) > 0 and np.nanmax(flat) > 0:
        top_idx = np.argsort(flat)[-top_n_hotspots:]
        hotspot_coords = []
        nlon = len(lon_centers)
        for idx in top_idx:
            y_idx = idx // nlon
            x_idx = idx % nlon
            hotspot_coords.append((lon_centers[x_idx], lat_centers[y_idx], flat[idx]))

        # plot hotspots as markers
        lons = [h[0] for h in hotspot_coords]
        lats = [h[1] for h in hotspot_coords]
        sizes = [max(6, (h[2] / max(flat)) * 20) for h in hotspot_coords]
        fig.add_trace(go.Scatter(
            x=lons, y=lats, mode='markers+text',
            marker=dict(size=sizes, color='black', line=dict(width=1, color='yellow')),
            text=[f"Hotspot: {int(h[2])}" for h in hotspot_coords], textposition='top center',
            hoverinfo='text'
        ))

    fig.update_layout(
        title="Risk Heatmap (weighted by risk score)",
        xaxis_title="Longitude", yaxis_title="Latitude",
        paper_bgcolor='black', plot_bgcolor='black', font_color='white'
    )
    return fig

def build_risk_trend(close_df: pd.DataFrame, forecast_hours:int=24, smoothing_window:int=3):
    """
    Build a professional risk trend chart:
    - hourly aggregation of risk_score (sum)
    - EWMA smoothing and peak detection
    - simple linear forecast for next `forecast_hours` with confidence band
    """
    if close_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Predicted Risks")
        return fig

    # ensure sat_time is datetime index
    df = close_df.copy()
    df['sat_time'] = pd.to_datetime(df['sat_time'])
    df = df.set_index('sat_time').sort_index()

    # Hourly aggregation: total risk per hour
    hourly = df['risk_score'].resample('1H').sum().fillna(0)
    if len(hourly) < 2:
        # not enough points for smoothing/forecast
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hourly.index, y=hourly.values, name='Hourly risk'))
        fig.update_layout(title="Not enough data for trend/forecast", paper_bgcolor='black', font_color='white')
        return fig

    # Smoothed series (EWMA)
    smoothed = hourly.ewm(alpha=0.3).mean()

    # Peak detection on smoothed series
    values = smoothed.values
    peaks, props = find_peaks(values, prominence=np.max(values)*0.15)
    peak_times = smoothed.index[peaks]
    peak_values = values[peaks]

    # Simple linear forecast on smoothed series (use timestamps as seconds)
    x_train = (smoothed.index.astype('int64') // 10**9).astype(float)
    y_train = values
    # fit linear model
    coef = np.polyfit(x_train, y_train, deg=1)
    poly = np.poly1d(coef)

    # forecast horizon
    last_ts = x_train[-1]
    future_x = np.arange(last_ts + 3600, last_ts + (forecast_hours+1)*3600, 3600)
    y_forecast = poly(future_x)

    # estimate residual std for simple CI
    residuals = y_train - poly(x_train)
    res_std = np.std(residuals) if len(residuals) > 1 else 0.0
    ci = 1.96 * res_std  # ~95% CI

    # Build figure with historical, smoothed, forecast and CI
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hourly.index, y=hourly.values, name='Hourly risk (raw)', marker_color='rgba(100,100,100,0.3)'))
    fig.add_trace(go.Scatter(x=smoothed.index, y=smoothed.values, mode='lines+markers', name='Smoothed risk', line=dict(color='orange', width=3)))
    # forecast line
    future_times = pd.to_datetime(future_x * 10**9)  # convert back to ns
    fig.add_trace(go.Scatter(x=future_times, y=y_forecast, mode='lines', name=f'Forecast next {forecast_hours}h', line=dict(color='red', dash='dash')))

    # add confidence band around forecast
    upper = y_forecast + ci
    lower = y_forecast - ci
    fig.add_trace(go.Scatter(
        x=list(future_times) + list(future_times[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill='toself', fillcolor='rgba(255,0,0,0.15)', line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip', showlegend=True, name='Forecast 95% CI'
    ))

    # annotate peaks
    for t,v in zip(peak_times, peak_values):
        fig.add_annotation(x=t, y=v, text=f"Peak {v:.1f}", showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color='yellow'))

    fig.update_layout(title="Predicted Risk Over Time (smoothed + forecast)", xaxis_title="Time", yaxis_title="Risk (sum)", paper_bgcolor='black', font_color='white')
    return fig

def map_close_to_coords(close_df: pd.DataFrame, df_positions: pd.DataFrame, time_tol_s:int=60) -> pd.DataFrame:
    """
    Map each close-approach row to the debris lat/lon by matching debris_name and nearest time.
    - close_df must have columns: ['sat_time','debris_name','risk_score', ...]
    - df_positions must have rows for many times with columns ['name','time','lat','lon'] (eg. df_all)
    Returns close_df augmented with 'lat' and 'lon' (NaN if not found).
    """
    if close_df.empty or df_positions.empty:
        return close_df.copy()

    pos = df_positions.copy()
    pos['time_bucket'] = pos['time'].dt.round(f'{DEFAULT_TIMESTEP_SEC}s')  # rounding to timestep
    close = close_df.copy()
    close['time_bucket'] = close['sat_time'].dt.round(f'{DEFAULT_TIMESTEP_SEC}s')

    # Merge on debris name + bucket
    merged = pd.merge(
        close, pos[['name','time_bucket','lat','lon']],
        left_on=['debris_name','time_bucket'], right_on=['name','time_bucket'],
        how='left', suffixes=('','_pos')
    )

    # For any rows where lat is NaN, try nearest-time lookup within time_tol_s
    missing = merged[merged['lat'].isna()].index
    if len(missing) > 0:
        # index pos by name for quick search
        pos_by_name = {n: g.sort_values('time') for n,g in pos.groupby('name')}
        for idx in missing:
            dname = merged.at[idx, 'debris_name']
            target_time = merged.at[idx, 'sat_time']
            if dname in pos_by_name:
                grp = pos_by_name[dname]
                # find absolute time difference
                times = grp['time'].values
                # compute nearest index
                diffs = np.abs((times - np.datetime64(target_time)).astype('timedelta64[s]').astype(int))
                min_i = int(np.argmin(diffs))
                if diffs[min_i] <= time_tol_s:
                    merged.at[idx, 'lat'] = grp.iloc[min_i]['lat']
                    merged.at[idx, 'lon'] = grp.iloc[min_i]['lon']
    # final: drop helper columns
    merged = merged.drop(columns=['name','time_bucket'], errors='ignore')
    return merged

def build_textured_earth(df_snap, title="Textured Earth with Debris"):
    """
    Build a 3D textured Earth globe with debris points.
    Uses Plotly surface for Earth texture + scatter3d for debris.
    """
    # Earth sphere mesh
    lons = np.linspace(-180, 180, 180)
    lats = np.linspace(-90, 90, 90)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    R = 6371  # Earth radius in km
    x = R * np.cos(np.radians(lat_grid)) * np.cos(np.radians(lon_grid))
    y = R * np.cos(np.radians(lat_grid)) * np.sin(np.radians(lon_grid))
    z = R * np.sin(np.radians(lat_grid))

    # Earth surface
    earth_surface = go.Surface(
        x=x, y=y, z=z,
        colorscale="Earth",
        showscale=False,
        opacity=0.8
    )

    # Debris scatter (use ECI x,y,z)
    debris_scatter = go.Scatter3d(
        x=df_snap["x"], y=df_snap["y"], z=df_snap["z"],
        mode="markers",
        marker=dict(size=2, color="cyan"),
        name="Debris"
    )

    fig = go.Figure(data=[earth_surface, debris_scatter])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
            aspectmode="data"
        ),
        paper_bgcolor="black",
        font_color="white"
    )
    return fig




# ----------------------------
# DASHBOARD
# ----------------------------
import io
import base64
from dash.dependencies import Input, Output
from dash import Dash, dcc, html
import plotly.graph_objects as go

def run_dashboard(df_all, df_clustered, close_df):
    app = Dash(__name__)

    # === Precompute cluster stats ===
    cluster_stats = []
    if not df_clustered.empty and "cluster" in df_clustered.columns:
        for cid, group in df_clustered.groupby("cluster"):
            if cid == -1:
                continue
            cluster_debris = group["name"].unique()
            cluster_risks = close_df[close_df["debris_name"].isin(cluster_debris)]

            if not cluster_risks.empty:
                avg_risk = cluster_risks["risk_score"].mean()
            else:
                avg_risk = 0  # no risks detected for this cluster

            # Emoji category
            if avg_risk < 20:
                risk_cat = "🟢 Low"
            elif avg_risk < 60:
                risk_cat = "🟡 Medium"
            else:
                risk_cat = "🔴 High"

            cluster_stats.append(
                f"Cluster {cid} → {len(group)} objects | Avg Risk: {avg_risk:.2f} | {risk_cat}"
            )

    # === Precompute predictive alerts ===
    alerts = []
    if not close_df.empty:
        for _, row in close_df.head(5).iterrows():
            prob = min(100, 100 * np.exp(-row['distance_km'] / 5) * (row['relative_velocity'] / 7))
            category = "🟢 Safe" if prob < 1 else "🟡 Caution" if prob < 5 else "🔴 High Risk"
            alerts.append(
                f"🚨 {row['debris_name']} | Dist={row['distance_km']} km | "
                f"V={row['relative_velocity']} km/s | P={prob:.2f}% | {category}"
            )

    # === Layout ===
    app.layout = html.Div(style={
        "background": "linear-gradient(to right, #0f2027, #203a43, #2c5364)",
        "color": "white",
        "fontFamily": "Segoe UI, sans-serif",
        "minHeight": "100vh",
        "display": "flex",
        "flexDirection": "column"
    }, children=[

        # Header Branding
        html.Div(style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "padding": "20px",
            "backgroundColor": "rgba(0,0,0,0.6)",
            "boxShadow": "0 4px 15px rgba(0,0,0,0.5)"
        }, children=[
            html.H1("🚀 OrbitalSight", style={"color": "#00eaff", "margin": "0"}),
            html.Div(children=[
                html.H3("ASTRA Hackathon 2025 | DTU", style={"margin": "0", "color": "#f1c40f"}),
                html.H4("Team: MindFirst", style={"margin": "0", "color": "#bdc3c7"})
            ])
        ]),

        # Alerts Section
        html.Div(style={
            "backgroundColor": "rgba(231,76,60,0.2)",
            "padding": "10px",
            "margin": "10px",
            "borderRadius": "8px"
        }, children=[
            html.H3("⚠️ Predictive Alerts", style={"color": "#e74c3c"}),
            html.Ul([html.Li(a) for a in alerts]) if alerts else html.P("No high-risk events detected.")
        ]),

        # Main Tabs
        html.Div(style={"flex": "1", "padding": "20px"}, children=[
            dcc.Tabs([

                dcc.Tab(label="🌍 3D Scatter", children=[
                    dcc.Graph(figure=build_3d_scatter(df_clustered))
                ]),

                dcc.Tab(label="🌐 Globe View", children=[
                    dcc.Graph(figure=build_globe(df_clustered))
                ]),

                dcc.Tab(label="🌍 Textured Earth View", children=[
                    dcc.Graph(figure=build_textured_earth(df_clustered))
                ]),

                dcc.Tab(label="🎥 Time-lapse", children=[
                    dcc.Graph(figure=build_orbit_animation(df_all))
                ]),

                dcc.Tab(label="🚨 Close Approaches", children=[
                    html.H3("Risk Table", style={"color": "#e74c3c"}),
                    html.Pre(close_df.head(20).to_string(index=False))
                ]),

                dcc.Tab(label="🔥 Risk Zones", children=[
                    dcc.Graph(figure=build_risk_heatmap(close_df, df_all))
                ]),

                dcc.Tab(label="📈 Risk Trend", children=[
                    dcc.Graph(figure=build_risk_trend(close_df))
                ]),

                dcc.Tab(label="📊 Cluster Insights", children=[
                    html.H3("Cluster Statistics", style={"color": "#00eaff"}),
                    html.Ul([html.Li(cs) for cs in cluster_stats]) if cluster_stats else html.P("No clusters detected.")
                ]),

                dcc.Tab(label="📥 Reports", children=[
                    html.H3("Download Reports", style={"color": "#f39c12"}),
                    html.Button("📄 Download CSV Report", id="btn_csv"),
                    dcc.Download(id="download_csv")
                ]),

                dcc.Tab(label="🚀 Future Scope", children=[
                    html.H3("Planned Extensions", style={"color": "#00eaff"}),
                    html.Ul([
                        html.Li("Integration with ISRO / NASA SSA systems"),
                        html.Li("ML models (LSTM/Transformers) for orbit forecasting"),
                        html.Li("Collision probability estimation from covariance data"),
                        html.Li("Crew safety extensions for human missions")
                    ])
                ])
            ])
        ]),

        # Footer Branding
        html.Footer("🔧 Built with ❤️ at ASTRA Hackathon 2025 | DTU | Team: MindFirst",
                    style={
                        "textAlign": "center",
                        "padding": "10px",
                        "fontSize": "14px",
                        "color": "#bdc3c7",
                        "backgroundColor": "rgba(0,0,0,0.6)",
                        "marginTop": "auto"
                    })
    ])

    # === CSV Download Callback ===
    @app.callback(
        Output("download_csv", "data"),
        Input("btn_csv", "n_clicks"),
        prevent_initial_call=True
    )
    def download_report(n_clicks):
        return dcc.send_data_frame(close_df.to_csv, "orbital_risk_report.csv")

    app.run(debug=False, port=8050)

    





# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    logging.info("Starting OrbitalSight...")

    # Step 1: Load TLEs
    tles = load_tles(CELESTRAK_URL, FALLBACK_FILE)
    epoch = datetime.now(UTC)

    # Step 2: Propagate
    df_all = propagate_all(tles, epoch, duration_hours=PROPAGATION_DURATION_HOURS)

    # Step 3: Snapshot + Clustering
    df_snap = snapshot(df_all, epoch)
    df_clustered = cluster_snapshot(df_snap)

    # Step 4: Close approaches
    sat_df = propagate_single_tle(tles[0], epoch, duration_hours=PREDICTION_HOURS)
    close_df = detect_close_approaches(df_all, sat_df)

    # Step 5: Dashboard
    run_dashboard(df_all, df_clustered, close_df)
