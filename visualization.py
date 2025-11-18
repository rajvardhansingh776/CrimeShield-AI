# dashboard.py
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
import json
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import networkx as nx
from pathlib import Path
import smtplib
from email.message import EmailMessage

st.set_page_config(layout="wide", page_title="Crime Hotspot Dashboard")

# ---------------------------
# CONFIG - adapt these paths
# ---------------------------
HOTSPOT_GEOJSON = "hotspots.geojson"        # produced by the forecasting pipeline (GeoJSON)
FIR_GEOJSON = "fir_points.geojson"          # optional: recent FIR incidents to overlay
POLL_CSV = "incoming_incidents/"            # directory to poll for new incident CSVs (or single file)
ALERT_PROB_THRESHOLD = 0.7                  # probability threshold to mark emerging hotspot
ALERT_LOOKBACK_MIN = 60                     # minutes to consider for recent events
EMAIL_ALERTS = False                         # flip True and configure email to send alerts

# ---------------------------
# HELPERS
# ---------------------------
@st.cache_data(ttl=60)
def load_hotspots(path):
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    return g

@st.cache_data(ttl=60)
def load_firs(path):
    try:
        g = gpd.read_file(path)
        if g.crs is None:
            g = g.set_crs("EPSG:4326")
        return g
    except Exception:
        return gpd.GeoDataFrame(columns=["geometry"])

def make_choropleth_map(grid_gdf, fir_gdf=None):
    # center map
    if grid_gdf.empty:
        m = folium.Map(location=[0,0], zoom_start=2)
        return m
    centroid = grid_gdf.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=13, control_scale=True)
    # color ramp by combined_prob_24h
    vals = grid_gdf["combined_prob_24h"].fillna(0).astype(float)
    vmin, vmax = vals.min(), vals.max()
    import branca.colormap as cm
    colormap = cm.LinearColormap(["green","yellow","red"], vmin=vmin, vmax=vmax).to_step(index=[vmin, (vmin+vmax)/2, vmax])
    # add polygons
    for _, row in grid_gdf.iterrows():
        prob = float(row.get("combined_prob_24h", 0.0) or 0.0)
        popup_html = f"""
        <b>Cell:</b> {row.get('cell_id')}<br/>
        <b>Prob (24h):</b> {prob:.3f}<br/>
        <b>Click for details in dashboard panel</b>
        """
        folium.GeoJson(
            row.geometry,
            style_function=lambda feature, p=prob: {
                "fillColor": colormap(p),
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.6
            },
            tooltip=f"cell:{row.get('cell_id')} prob:{prob:.3f}",
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)
    colormap.caption = "Predicted Crime Probability (24h)"
    colormap.add_to(m)
    # overlay FIR points if provided
    if fir_gdf is not None and not fir_gdf.empty:
        for _, r in fir_gdf.iterrows():
            if r.geometry is None or r.geometry.is_empty:
                continue
            folium.CircleMarker(location=[r.geometry.y, r.geometry.x], radius=3, color="blue", fill=True, opacity=0.8).add_to(m)
    return m

def compute_feature_contributions(rf_model, scaler, sample_df, feature_cols, topk=6):
    """
    If RandomForest available: compute permutation importance on local sample.
    Returns a sorted list of (feature, importance, local_value)
    """
    try:
        # scale sample
        X = sample_df[feature_cols].fillna(0)
        if scaler is not None:
            Xs = scaler.transform(X)
        else:
            Xs = X.values
        # For robust local importance we use permutation_importance on the model using sample repeated draws
        r = permutation_importance(rf_model, Xs, sample_df["_label_for_perm"] if "_label_for_perm" in sample_df.columns else np.zeros(len(Xs)), n_repeats=10, random_state=0, n_jobs=1)
        importances = r.importances_mean
        feats = list(feature_cols)
        items = sorted(zip(feats, importances, X.iloc[0].values), key=lambda x: x[1], reverse=True)[:topk]
        return items
    except Exception as e:
        # fallback: use columns with highest absolute z-score from population (heuristic)
        vals = sample_df[feature_cols].iloc[0]
        z = (vals - sample_df[feature_cols].mean()) / (sample_df[feature_cols].std() + 1e-9)
        items = sorted(zip(feature_cols, z.abs(), vals), key=lambda x: x[1], reverse=True)[:topk]
        return items

def nearest_neighbor_route(points_gdf):
    # basic nearest neighbor TSP heuristic returning ordered list of lat-lon
    pts = points_gdf.copy()
    pts = pts.to_crs(epsg=3857)
    coords = [(i, pt.geometry.x, pt.geometry.y) for i,pt in pts.iterrows()]
    if not coords:
        return []
    visited = []
    remaining = coords[:]
    cur = remaining.pop(0)
    visited.append(cur)
    while remaining:
        dists = [((cur[1]-r[1])**2 + (cur[2]-r[2])**2, r) for r in remaining]
        best = min(dists, key=lambda x:x[0])[1]
        visited.append(best)
        remaining.remove(best)
        cur = best
    # convert back to latlon
    points_gdf = points_gdf.to_crs(epsg=4326)
    ordered = []
    for v in visited:
        # v[0] is original index
        idx = v[0]
        row = points_gdf.loc[idx]
        ordered.append((row.geometry.y, row.geometry.x))
    return ordered

def send_email_alert(to_address, subject, body, smtp_config):
    # smtp_config = dict(server, port, username, password, from)
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = smtp_config["from"]
        msg["To"] = to_address
        msg.set_content(body)
        with smtplib.SMTP(smtp_config["server"], smtp_config["port"]) as s:
            s.starttls()
            s.login(smtp_config["username"], smtp_config["password"])
            s.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email alert failed: {e}")
        return False

# ---------------------------
# Dashboard layout
# ---------------------------
st.title("Crime Hotspot Dashboard — Visualization & Actionability")
left, right = st.columns([2,1])

with right:
    st.header("Controls")
    hotspot_file = st.text_input("Hotspot GeoJSON path", HOTSPOT_GEOJSON)
    fir_file = st.text_input("FIR GeoJSON (optional)", FIR_GEOJSON)
    st.write("Alert settings")
    prob_threshold = st.slider("Alert probability threshold", 0.0, 1.0, ALERT_PROB_THRESHOLD)
    refresh = st.button("Refresh Now")
    st.write("---")
    st.markdown("**Alert log**")
    alert_log = st.empty()
    st.write("---")
    st.markdown("Export / actions")
    export_geojson = st.button("Export current hotspots GeoJSON")
    if export_geojson:
        try:
            g = load_hotspots(hotspot_file)
            g.to_file("hotspots_exported.geojson", driver="GeoJSON")
            st.success("Exported to hotspots_exported.geojson")
        except Exception as e:
            st.error(f"Export failed: {e}")

with left:
    # load data
    try:
        grid = load_hotspots(hotspot_file)
    except Exception as e:
        st.error(f"Failed to load hotspot file: {e}")
        grid = gpd.GeoDataFrame(columns=["geometry"])
    firs = load_firs(fir_file) if fir_file else gpd.GeoDataFrame(columns=["geometry"])

    st.subheader("Crime Heat Map")
    m = make_choropleth_map(grid, firs)
    map_result = st_folium(m, width=900, height=600)

    # click handling - st_folium returns last_clicked
    clicked = map_result.get("last_clicked")
    selected_cell_id = None
    if clicked:
        # last_clicked: {"lat":.., "lng":..} - find polygon containing that point
        pt = Point(clicked["lng"], clicked["lat"])
        sel = grid[grid.geometry.contains(pt)]
        if not sel.empty:
            selected_cell_id = sel.iloc[0]["cell_id"]
            st.success(f"Selected cell: {selected_cell_id}")
        else:
            st.info("Clicked outside cells or cell not found.")

    st.write("---")
    st.subheader("Hotspot summary")
    if grid.empty:
        st.write("No hotspot data loaded.")
    else:
        topk = st.slider("Top K hotspots to list", 3, 30, 10)
        df_view = grid[['cell_id','combined_prob_24h']].fillna(0).sort_values("combined_prob_24h", ascending=False).head(topk)
        st.table(df_view.assign(combined_prob_24h=lambda d: d["combined_prob_24h"].map(lambda x: f"{x:.3f}")))

    # Drilldown panel
    st.write("---")
    st.subheader("Drill-down: Selected cell details")
    if selected_cell_id is None:
        st.info("Click a cell on the map to see predictive factors and recommended actions.")
    else:
        sel_row = grid[grid["cell_id"]==selected_cell_id].iloc[0]
        st.markdown(f"**Cell:** {selected_cell_id}")
        st.markdown(f"**Predicted Prob (24h):** {sel_row.get('combined_prob_24h',0):.3f}")
        # collect feature columns heuristically
        reserved = {"cell_id","geometry","combined_prob_24h","rf_prob_24h","kde_score","lstm_prob_24h"}
        feature_cols = [c for c in grid.columns if c not in reserved]
        # prepare one-row DF for explanation
        sample_df = pd.DataFrame([sel_row.drop(labels=[c for c in sel_row.index if c == "geometry"])])
        # If the RF model and scaler exist in the grid (rare) try to use them; otherwise expect external model object
        rf_model = None
        rf_scaler = None
        if "_rf_model" in st.session_state:
            rf_model = st.session_state["_rf_model"]
        if "_rf_scaler" in st.session_state:
            rf_scaler = st.session_state["_rf_scaler"]
        st.markdown("**Top contributing factors (heuristic / permutation importance):**")
        contribs = compute_feature_contributions(rf_model, rf_scaler, sample_df, feature_cols, topk=6)
        rows = []
        for f,imp,val in contribs:
            rows.append({"feature":f, "importance":float(imp), "local_value":float(val)})
        st.table(pd.DataFrame(rows).assign(importance=lambda d: d["importance"].map(lambda x: f"{x:.4f}"),
                                           local_value=lambda d: d["local_value"].map(lambda x: f"{x:.3f}")))
        st.write("---")
        st.markdown("**Local values (all features)**")
        st.dataframe(sample_df[feature_cols].T.rename(columns={0:"value"}))

        # suggested patrol route: choose top N nearby hotspot cells
        st.write("---")
        st.markdown("**Suggested quick patrol route across top nearby hotspots**")
        # find top K cells ordered by distance to selected centroid
        sel_centroid = sel_row.geometry.centroid
        grid_copy = grid.copy()
        grid_copy["dist_m"] = grid_copy.geometry.centroid.to_crs(epsg=3857).distance(gpd.GeoSeries([sel_centroid]).to_crs(epsg=3857)[0])
        candidates = grid_copy.sort_values(["dist_m","combined_prob_24h"], ascending=[True,False]).head(8)
        route_pts = candidates.reset_index()
        route_order = nearest_neighbor_route(route_pts)
        if route_order:
            st.markdown("Route waypoints (lat,lon):")
            for i,pt in enumerate(route_order):
                st.write(f"{i+1}. {pt[0]:.6f}, {pt[1]:.6f}")
            # draw route on map: create a small new map and show polyline
            route_map = folium.Map(location=[sel_centroid.y, sel_centroid.x], zoom_start=14)
            folium.PolyLine(route_order, weight=4, opacity=0.7).add_to(route_map)
            for i,loc in enumerate(route_order):
                folium.Marker(location=loc, popup=f"Stop {i+1}").add_to(route_map)
            st_folium(route_map, height=350)
        else:
            st.write("No route available (not enough points).")

# ---------------------------
# ALERTING (simple polling)
# ---------------------------

st.sidebar.header("Real-time alerting (polling)")
poll_enable = st.sidebar.checkbox("Enable polling for new incidents", value=False)
poll_interval = st.sidebar.slider("Poll every (seconds)", 10, 600, 60)
last_polled = st.session_state.get("_last_polled_time", None)
alert_history = st.session_state.get("_alert_history", [])

def poll_for_new_incidents_and_alert():
    global alert_history
    p = Path(POLL_CSV)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(list(p.glob("*.csv")), key=lambda x: x.stat().st_mtime, reverse=True)
    else:
        files = []
    new_alerts = []
    for f in files[:5]:
        try:
            df = pd.read_csv(f)
            if "date_time" in df.columns and ("latitude" in df.columns or "lat" in df.columns):
                df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
                df["latitude"] = df.get("latitude", df.get("lat", np.nan))
                df["longitude"] = df.get("longitude", df.get("lon", df.get("longitude", np.nan)))
                recent_cut = datetime.utcnow() - timedelta(minutes=ALERT_LOOKBACK_MIN)
                recents = df[df["date_time"] >= recent_cut]
                if recents.empty:
                    continue
                # map recents to grid
                pts = gpd.GeoDataFrame(recents, geometry=gpd.points_from_xy(recents["longitude"], recents["latitude"]), crs="EPSG:4326")
                # spatial join
                if not grid.empty:
                    joined = gpd.sjoin(pts, grid, how="left", predicate="within")
                    # find any cell with prob >= threshold
                    cand = joined.groupby("cell_id").size().reset_index(name="n_incidents").dropna(subset=["cell_id"])
                    for _, r in cand.iterrows():
                        cellid = r["cell_id"]
                        cellrow = grid[grid["cell_id"]==cellid].iloc[0]
                        prob = float(cellrow.get("combined_prob_24h",0.0) or 0.0)
                        if prob >= prob_threshold:
                            alert_msg = f"[{datetime.utcnow().isoformat()}] Emerging hotspot {cellid} prob={prob:.3f} recent_events={int(r['n_incidents'])}"
                            new_alerts.append({"cell":cellid, "prob":prob, "msg":alert_msg})
        except Exception as e:
            st.warning(f"Failed reading {f}: {e}")
    # deduplicate and log
    for a in new_alerts:
        if a["msg"] not in alert_history:
            alert_history.append(a["msg"])
            # on-screen
            st.toast(a["msg"]) if hasattr(st, "toast") else st.info(a["msg"])
            if EMAIL_ALERTS:
                # configure your smtp settings here if you want to send emails
                smtp_cfg = {"server":"smtp.example.com","port":587,"username":"user","password":"pass","from":"alerts@example.com"}
                send_email_alert("ops-team@example.com", f"Hotspot Alert: {a['cell']}", a["msg"], smtp_cfg)
    # update session state
    st.session_state["_alert_history"] = alert_history
    st.session_state["_last_polled_time"] = datetime.utcnow().isoformat()

# run polling if enabled
if poll_enable:
    # simple blocking poll triggered by manual refresh or timer
    if refresh:
        poll_for_new_incidents_and_alert()
    else:
        # Non-blocking: show next poll time and allow manual poll
        last = st.session_state.get("_last_polled_time", "never")
        st.sidebar.write(f"Last polled: {last}")
        if st.sidebar.button("Poll now"):
            poll_for_new_incidents_and_alert()
        # auto-refresh the page after poll_interval seconds is possible using st.experimental_rerun, but avoid aggressive reruns
        if "next_poll" not in st.session_state:
            st.session_state["next_poll"] = time.time() + poll_interval
        if time.time() >= st.session_state["next_poll"]:
            poll_for_new_incidents_and_alert()
            st.session_state["next_poll"] = time.time() + poll_interval

# show alert log
if "_alert_history" in st.session_state:
    st.sidebar.write("\n".join(st.session_state["_alert_history"][-20:]))