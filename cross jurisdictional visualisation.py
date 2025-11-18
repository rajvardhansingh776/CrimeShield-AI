# cross_jurisdiction_viz.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
import networkx as nx
from pyvis.network import Network
import folium
from streamlit_folium import st_folium
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

st.set_page_config(layout="wide", page_title="Cross-Jurisdictional Network Visualizer")

# -----------------------
# Helpers
# -----------------------
def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".geojson", ".json", ".gpkg", ".shp"]:
        return gpd.read_file(path)
    else:
        return pd.read_csv(path)

def ensure_gdf_from_fir(df, source_label="src"):
    df = df.copy()
    # normalize expected columns
    if "date_time" not in df.columns:
        for c in ["date","datetime","reported_at","incident_time"]:
            if c in df.columns:
                df["date_time"] = df[c]; break
    if "latitude" not in df.columns and "lat" in df.columns:
        df["latitude"] = df["lat"]
    if "longitude" not in df.columns and "lon" in df.columns:
        df["longitude"] = df["lon"]
    if "fir_id" not in df.columns:
        df["fir_id"] = df.index.astype(str)
    if "crime_type" not in df.columns:
        df["crime_type"] = df.get("offence", "unknown")
    if "accused_list" not in df.columns:
        # try several common names
        for c in ["accused","accused_party","accused_list","persons"]:
            if c in df.columns:
                df["accused_list"] = df[c]; break
    # convert to gdf
    if not isinstance(df, gpd.GeoDataFrame):
        if ("longitude" in df.columns) and ("latitude" in df.columns):
            df['geometry'] = df.apply(lambda r: Point(float(r['longitude']), float(r['latitude'])) if pd.notna(r['longitude']) and pd.notna(r['latitude']) else None, axis=1)
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        else:
            gdf = gpd.GeoDataFrame(df, geometry=None)
    else:
        gdf = df
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
    gdf["source"] = source_label
    # parse datetimes
    if "date_time" in gdf.columns:
        gdf["date_time"] = pd.to_datetime(gdf["date_time"], errors="coerce")
    else:
        gdf["date_time"] = pd.NaT
    return gdf

def ensure_gdf_from_cdr(df, source_label="src"):
    df = df.copy()
    if "start_time" not in df.columns:
        for c in ["date_time","start","call_time","timestamp"]:
            if c in df.columns:
                df["start_time"] = df[c]; break
    if "tower_lat" not in df.columns and ("latitude" in df.columns or "lat" in df.columns):
        df["tower_lat"] = df.get("latitude", df.get("lat"))
    if "tower_lon" not in df.columns and ("longitude" in df.columns or "lon" in df.columns):
        df["tower_lon"] = df.get("longitude", df.get("lon"))
    if "cdr_id" not in df.columns:
        df["cdr_id"] = df.index.astype(str)
    if not isinstance(df, gpd.GeoDataFrame):
        if ("tower_lon" in df.columns) and ("tower_lat" in df.columns):
            df['geometry'] = df.apply(lambda r: Point(float(r['tower_lon']), float(r['tower_lat'])) if pd.notna(r['tower_lon']) and pd.notna(r['tower_lat']) else None, axis=1)
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        else:
            gdf = gpd.GeoDataFrame(df, geometry=None)
    else:
        gdf = df
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
    gdf["source"] = source_label
    if "start_time" in gdf.columns:
        gdf["start_time"] = pd.to_datetime(gdf["start_time"], errors="coerce")
    else:
        gdf["start_time"] = pd.NaT
    return gdf

def build_graph_from_filtered(fir_gdfs, cdr_gdfs, proximity_m=100):
    G = nx.Graph()
    # Cases and accused
    for gdf in fir_gdfs:
        src = gdf["source"].iloc[0] if "source" in gdf.columns else "FIR"
        for _, r in gdf.iterrows():
            cid = f"Case::{src}::{r.get('fir_id')}"
            G.add_node(cid, label="Case", source=src, crime_type=str(r.get("crime_type","")), date_time=str(r.get("date_time","")))
            # accused parsing
            accused_raw = r.get("accused_list")
            if pd.notna(accused_raw):
                if isinstance(accused_raw, str):
                    items = [a.strip() for a in accused_raw.split(";") if a.strip()]
                elif isinstance(accused_raw, (list, tuple, set)):
                    items = list(accused_raw)
                else:
                    items = [str(accused_raw)]
                for a in items:
                    pid = f"Person::{src}::{a[:60]}"
                    G.add_node(pid, label="Person", source=src, raw=a)
                    G.add_edge(pid, cid, relation="CO_OFFEND")
    # CDR -> phones and calls
    for gdf in cdr_gdfs:
        src = gdf["source"].iloc[0] if "source" in gdf.columns else "CDR"
        for _, r in gdf.iterrows():
            caller = r.get("caller_id")
            receiver = r.get("receiver_id")
            if pd.isna(caller) or pd.isna(receiver):
                continue
            a = f"Phone::{src}::{str(caller)[:60]}"
            b = f"Phone::{src}::{str(receiver)[:60]}"
            G.add_node(a, label="Phone", source=src, raw=str(caller))
            G.add_node(b, label="Phone", source=src, raw=str(receiver))
            G.add_edge(a, b, relation="CALLED", start_time=str(r.get("start_time","")), duration=float(r.get("duration_seconds") or 0))
    # Optional location proximity (link cases to nearby phone towers / locations)
    # Collect case points and tower points
    case_points = []
    for gdf in fir_gdfs:
        if gdf.geometry is not None:
            pts = gdf[~gdf.geometry.isna()][["fir_id","geometry","source"]]
            for _,r in pts.iterrows():
                case_points.append((f"Case::{r['source']}::{r['fir_id']}", r["geometry"]))
    tower_points = []
    for gdf in cdr_gdfs:
        if gdf.geometry is not None:
            pts = gdf[~gdf.geometry.isna()][["cdr_id","geometry","source"]]
            for _,r in pts.iterrows():
                tower_points.append((f"Tower::{r['source']}::{r['cdr_id']}", r["geometry"]))
    # create proximity edges
    for cid, cgeom in case_points:
        if cgeom is None:
            continue
        for tid, tgeom in tower_points:
            if tgeom is None:
                continue
            if cgeom.distance(tgeom) * 111000.0 <= proximity_m:  # approx conversion if in degrees (coarse)
                G.add_node(tid, label="Tower")
                G.add_edge(cid, tid, relation="NEAR_TOWER")
    return G

def pyvis_from_networkx(G, notebook=False, height="600px", width="100%"):
    net = Network(height=height, width=width, notebook=notebook, bgcolor="#ffffff", font_color="#222222")
    net.force_atlas_2based()
    # style nodes by label/source and centrality
    deg = dict(G.degree())
    for n, attrs in G.nodes(data=True):
        label = attrs.get("label", n)
        title = "<br/>".join([f"{k}:{v}" for k,v in attrs.items()])
        size = 5 + np.log1p(deg.get(n,0))*6
        color = "#97c2fc"
        if attrs.get("label") == "Person":
            color = "#ffcc00"
        elif attrs.get("label") == "Phone":
            color = "#ff6666"
        elif attrs.get("label") == "Case":
            color = "#66cc66"
        net.add_node(n, label=str(label), title=title, size=size, color=color)
    for u,v,edata in G.edges(data=True):
        rel = edata.get("relation", "")
        width = 1 + (1 if rel=="CALLED" else 0)
        net.add_edge(u, v, title=rel, width=width)
    return net

def embed_pyvis(net, height=600):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    path = tmp.name
    net.show(path)
    html = open(path, 'r', encoding='utf-8').read()
    return html, path

# -----------------------
# UI: Upload files
# -----------------------
st.title("Cross-Jurisdictional Network Visualizer")

st.sidebar.header("Upload data per jurisdiction")
uploaded = st.sidebar.file_uploader("Upload multiple FIR/CDR files (CSV/GeoJSON). For each file you upload, enter a label.", accept_multiple_files=True)
file_labels = {}
uploaded_files = uploaded or []
for up in uploaded_files:
    label = st.sidebar.text_input(f"Label for {up.name}", value=up.name, key=f"lbl_{up.name}")
    file_labels[up.name] = label

# allow user to load example (skip here) or local paths
st.sidebar.markdown("Or point to local folder containing jurisdiction files (CSV/GeoJSON)")
local_folder = st.sidebar.text_input("Local folder (optional)", "")

# -----------------------
# Load & normalize
# -----------------------
fir_gdfs = []
cdr_gdfs = []
if uploaded_files:
    for up in uploaded_files:
        # try to guess type by filename
        try:
            df = read_table(up)
        except Exception:
            try:
                df = pd.read_csv(up)
            except Exception:
                st.sidebar.error(f"Failed reading {up.name}")
                continue
        lbl = file_labels.get(up.name, up.name)
        # heuristics: if contains 'fir' or 'crime' assume FIR, else if 'caller' assume CDR
        cols = set(df.columns.str.lower())
        if any(x in cols for x in ("fir_id","crime_type","accused","offence")):
            gdf = ensure_gdf_from_fir(df, source_label=lbl)
            fir_gdfs.append(gdf)
        elif any(x in cols for x in ("caller_id","receiver_id","start_time","cdr_id")):
            gdf = ensure_gdf_from_cdr(df, source_label=lbl)
            cdr_gdfs.append(gdf)
        else:
            # ask user what it is
            kind = st.sidebar.selectbox(f"Select type for {up.name}", ["FIR","CDR"], key=f"type_{up.name}")
            if kind=="FIR":
                gdf = ensure_gdf_from_fir(df, source_label=lbl)
                fir_gdfs.append(gdf)
            else:
                gdf = ensure_gdf_from_cdr(df, source_label=lbl)
                cdr_gdfs.append(gdf)

# load local folder files if specified
if local_folder:
    p = Path(local_folder)
    if p.exists() and p.is_dir():
        for fp in p.glob("*"):
            try:
                df = read_table(fp)
            except Exception:
                continue
            lbl = fp.stem
            cols = set(df.columns.str.lower())
            if any(x in cols for x in ("fir_id","crime_type","accused","offence")):
                fir_gdfs.append(ensure_gdf_from_fir(df, source_label=lbl))
            elif any(x in cols for x in ("caller_id","receiver_id","start_time","cdr_id")):
                cdr_gdfs.append(ensure_gdf_from_cdr(df, source_label=lbl))
            else:
                # default to FIR
                fir_gdfs.append(ensure_gdf_from_fir(df, source_label=lbl))

# if nothing uploaded, show message
if not fir_gdfs and not cdr_gdfs:
    st.info("Upload or point to FIR/CDR files in the sidebar to begin.")
    st.stop()

# -----------------------
# Filters
# -----------------------
st.sidebar.header("Filters")
all_crime_types = sorted({ct for g in fir_gdfs for ct in g.get("crime_type", pd.Series([])).dropna().unique()})
crime_sel = st.sidebar.multiselect("Crime type (across jurisdictions)", options=all_crime_types, default=all_crime_types[:5] if all_crime_types else [])
min_date = min([g["date_time"].min() for g in fir_gdfs if "date_time" in g.columns and not g["date_time"].isna().all()] + [pd.NaT])
max_date = max([g["date_time"].max() for g in fir_gdfs if "date_time" in g.columns and not g["date_time"].isna().all()] + [pd.NaT])
date_range = st.sidebar.date_input("Date range (FIR)", [min_date.date() if pd.notna(min_date) else datetime.utcnow().date(), max_date.date() if pd.notna(max_date) else datetime.utcnow().date()])
min_degree = st.sidebar.slider("Minimum node degree (network)", 0, 10, 0)
proximity_m = st.sidebar.number_input("Proximity (meters) for linking cases->towers (approx)", value=200)

# geographic bbox input (minx, miny, maxx, maxy)
st.sidebar.markdown("Filter by geographic bbox (lon/lat). Leave blank to use all.")
bbox_vals = st.sidebar.text_input("bbox: minlon,minlat,maxlon,maxlat", value="")

# -----------------------
# Apply filters & build merged gdfs
# -----------------------
def apply_fir_filters(gdfs, crime_sel, date_range, bbox_vals):
    out = []
    for g in gdfs:
        gg = g.copy()
        if crime_sel:
            gg = gg[gg["crime_type"].isin(crime_sel)]
        if "date_time" in gg.columns and not gg["date_time"].isna().all():
            start = pd.to_datetime(date_range[0])
            end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
            gg = gg[(gg["date_time"] >= start) & (gg["date_time"] < end)]
        if bbox_vals:
            try:
                a,b,c,d = [float(x.strip()) for x in bbox_vals.split(",")]
                poly = box(a,b,c,d)
                gg = gg[~gg.geometry.isna()]
                gg = gg[gg.geometry.within(poly)]
            except Exception:
                pass
        out.append(gg)
    return out

def apply_cdr_filters(gdfs, date_range, bbox_vals):
    out = []
    for g in gdfs:
        gg = g.copy()
        if "start_time" in gg.columns and not gg["start_time"].isna().all():
            start = pd.to_datetime(date_range[0])
            end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
            gg = gg[(gg["start_time"] >= start) & (gg["start_time"] < end)]
        if bbox_vals:
            try:
                a,b,c,d = [float(x.strip()) for x in bbox_vals.split(",")]
                poly = box(a,b,c,d)
                gg = gg[~gg.geometry.isna()]
                gg = gg[gg.geometry.within(poly)]
            except Exception:
                pass
        out.append(gg)
    return out

fir_filtered = apply_fir_filters(fir_gdfs, crime_sel, date_range, bbox_vals)
cdr_filtered = apply_cdr_filters(cdr_gdfs, date_range, bbox_vals)

# -----------------------
# Build graph and compute centrality (light)
# -----------------------
G = build_graph_from_filtered(fir_filtered, cdr_filtered, proximity_m=proximity_m)
if G.number_of_nodes() == 0:
    st.warning("Filtered data has no nodes. Expand filters.")
    st.stop()

deg = dict(G.degree())
nx.set_node_attributes(G, deg, "degree")

# compute a couple of centralities (fast approximate)
try:
    pagerank = nx.pagerank(G)
except Exception:
    pagerank = {n:0.0 for n in G.nodes()}
nx.set_node_attributes(G, pagerank, "pagerank")

# -----------------------
# Layout: Map + Network
# -----------------------
left_col, right_col = st.columns([2,1])

with left_col:
    st.subheader("Geographic view")
    # assemble merged points for display
    merged_points = []
    for g in fir_filtered:
        if g is None: continue
        gp = g[~g.geometry.isna()][["fir_id","crime_type","date_time","source","geometry"]].copy()
        gp["type"] = "FIR"
        merged_points.append(gp)
    for g in cdr_filtered:
        if g is None: continue
        gp = g[~g.geometry.isna()][["cdr_id","start_time","source","geometry"]].copy()
        gp = gp.rename(columns={"start_time":"date_time","cdr_id":"id"})
        gp["type"] = "CDR"
        merged_points.append(gp)
    if merged_points:
        merged = gpd.GeoDataFrame(pd.concat(merged_points, ignore_index=True), crs="EPSG:4326")
    else:
        merged = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    if merged.empty:
        st.write("No geographic points to show.")
    else:
        centroid = merged.geometry.unary_union.centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11)
        for _, r in merged.iterrows():
            lat = r.geometry.y; lon = r.geometry.x
            popup = folium.Popup(f"Type: {r.get('type')}<br>Source: {r.get('source')}<br>Date: {r.get('date_time')}", max_width=300)
            color = "blue" if r.get("type")=="FIR" else "purple"
            folium.CircleMarker(location=[lat, lon], radius=4, color=color, fill=True, popup=popup).add_to(m)
        st_folium(m, width=900, height=600)

with right_col:
    st.subheader("Network view & filters")
    st.markdown(f"Nodes: **{G.number_of_nodes()}**, Edges: **{G.number_of_edges()}**")
    # allow filtering by min_degree
    nodes_keep = [n for n,d in G.degree() if d>=min_degree]
    subG = G.subgraph(nodes_keep).copy()
    st.write(f"Nodes after degree filter: {subG.number_of_nodes()}")
    # show top influencers by pagerank
    pr_sorted = sorted(subG.nodes(data=True), key=lambda x: x[1].get("pagerank",0), reverse=True)[:15]
    pr_df = pd.DataFrame([{"node":n, "label":d.get("label"), "pagerank":d.get("pagerank"), "degree":d.get("degree")} for n,d in pr_sorted])
    st.table(pr_df)

    st.markdown("### Interactive network (drag nodes). Hover to see attributes.")
    net = pyvis_from_networkx(subG, notebook=False, height="600px", width="100%")
    html, html_path = embed_pyvis(net, height=600)
    st.components.v1.html(html, height=620, scrolling=True)

    # download options
    st.markdown("### Export")
    if st.button("Export filtered FIR GeoJSON"):
        out_gdf = gpd.GeoDataFrame(pd.concat(fir_filtered, ignore_index=True), crs="EPSG:4326")
        out_path = "filtered_fir.geojson"
        out_gdf.to_file(out_path, driver="GeoJSON")
        st.success(f"Saved {out_path}")
    if st.button("Export filtered CDR GeoJSON"):
        out_gdf = gpd.GeoDataFrame(pd.concat(cdr_filtered, ignore_index=True), crs="EPSG:4326")
        out_path = "filtered_cdr.geojson"
        out_gdf.to_file(out_path, driver="GeoJSON")
        st.success(f"Saved {out_path}")
    if st.button("Export GraphML"):
        nx.write_graphml(subG, "filtered_graph.graphml")
        st.success("Saved filtered_graph.graphml")

st.sidebar.markdown("---")
st.sidebar.markdown("Notes: This is a prototype. For production you may want to:")
st.sidebar.markdown("- Use PostGIS or spatial indexing for faster geo queries across large jurisdictions.")
st.sidebar.markdown("- Use Neo4j / Neo4j Bloom for richer cross-jurisdiction graph exploration + precomputed GDS centralities.")
st.sidebar.markdown("- Add authentication / audit logging for investigative workflows.")
