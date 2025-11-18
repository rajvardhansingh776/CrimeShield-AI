"""
build_crime_graph.py

Build a crime knowledge graph from FIR, CDR and device/location inputs.

Modes:
- neo4j: push nodes & relationships to a running Neo4j instance via py2neo
- networkx: build a local NetworkX graph and optionally export GraphML/CSV for Neo4j import

Install:
pip install pandas py2neo networkx python-dateutil tqdm

Usage example:
from build_crime_graph import build_graph_from_files
build_graph_from_files(
    fir_path="data/fir.csv",
    cdr_path="data/cdr.csv",
    device_path="data/devices.csv",
    mode="neo4j",
    neo4j_config={"uri":"bolt://localhost:7687","user":"neo4j","password":"secret"}
)
"""

import pandas as pd
import networkx as nx
from datetime import datetime
from tqdm import tqdm
import hashlib
import os

# Try import py2neo but keep optional
try:
    from py2neo import Graph, Node, Relationship
    HAS_PY2NEO = True
except Exception:
    HAS_PY2NEO = False

# ---------- Utilities ----------
def anonymize_id(val, salt="graph_salt"):
    if pd.isna(val):
        return None
    return hashlib.sha256((str(val) + salt).encode()).hexdigest()[:20]

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

# ---------- Neo4j helpers ----------
def connect_neo4j(uri, user, password):
    if not HAS_PY2NEO:
        raise RuntimeError("py2neo not installed. Install with `pip install py2neo` to use Neo4j mode.")
    g = Graph(uri, auth=(user, password))
    return g

def create_neo4j_schema(graph):
    # Create uniqueness constraints (id-style)
    # Note: Neo4j versions differ on syntax; these cyphers are compatible with recent versions.
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Person) REQUIRE a.person_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Phone) REQUIRE p.phone_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Device) REQUIRE d.imei IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Case) REQUIRE f.case_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.loc_id IS UNIQUE"
    ]
    for c in constraints:
        try:
            graph.run(c)
        except Exception:
            # ignore failures (older versions)
            pass

# ---------- Graph building functions ----------
def ingest_fir_cases_to_neo4j(graph, fir_df, salt="graph_salt"):
    """Create Case nodes and Person nodes and CO_OFFEND relationships."""
    fir_df = fir_df.copy()
    ensure_cols(fir_df, ["fir_id","date_time","crime_type","latitude","longitude","accused_list","victim_list"])
    tx = graph.begin()
    for _, row in tqdm(fir_df.iterrows(), total=len(fir_df), desc="Ingesting FIRs to Neo4j"):
        case_id_raw = row["fir_id"]
        if pd.isna(case_id_raw):
            continue
        case_id = str(case_id_raw)
        case_node = Node("Case", case_id=case_id, crime_type=str(row.get("crime_type", "") or ""), date_time=str(row.get("date_time","")), latitude=row.get("latitude"), longitude=row.get("longitude"))
        tx.merge(case_node, "Case", "case_id")
        # Accused list expected as iterable string (e.g., "acc1;acc2") or actual list
        accused = row.get("accused_list")
        if pd.notna(accused):
            if isinstance(accused, str):
                accused_items = [a.strip() for a in accused.split(";") if a.strip()]
            elif isinstance(accused, (list, tuple, set)):
                accused_items = accused
            else:
                accused_items = [str(accused)]
            for a in accused_items:
                pid = anonymize_id(a, salt)
                pnode = Node("Person", person_id=pid, raw_id=str(a)[:60])
                tx.merge(pnode, "Person", "person_id")
                rel = Relationship(pnode, "CO_OFFEND", case_node)
                tx.merge(rel)
        # Victims (optional) - create Person nodes with role
        victims = row.get("victim_list")
        if pd.notna(victims):
            if isinstance(victims, str):
                vic_items = [v.strip() for v in victims.split(";") if v.strip()]
            elif isinstance(victims, (list, tuple, set)):
                vic_items = victims
            else:
                vic_items = [str(victims)]
            for v in vic_items:
                vid = anonymize_id(v, salt)
                vnode = Node("Person", person_id=vid, raw_id=str(v)[:60], role="victim")
                tx.merge(vnode, "Person", "person_id")
                r = Relationship(vnode, "INVOLVED_IN", case_node)
                tx.merge(r)
    tx.commit()

def ingest_cdr_to_neo4j(graph, cdr_df, salt="graph_salt"):
    """Create Phone nodes and CALL edges between them; optionally link to persons if mapping available."""
    cdr_df = cdr_df.copy()
    ensure_cols(cdr_df, ["cdr_id","caller_id","receiver_id","start_time","duration_seconds","tower_lat","tower_lon","imei"])
    tx = graph.begin()
    for _, row in tqdm(cdr_df.iterrows(), total=len(cdr_df), desc="Ingesting CDRs to Neo4j"):
        caller = row.get("caller_id")
        receiver = row.get("receiver_id")
        if pd.isna(caller) or pd.isna(receiver):
            continue
        caller_id = anonymize_id(caller, salt)
        receiver_id = anonymize_id(receiver, salt)
        caller_node = Node("Phone", phone_id=caller_id, raw_phone=str(caller)[:60])
        receiver_node = Node("Phone", phone_id=receiver_id, raw_phone=str(receiver)[:60])
        tx.merge(caller_node, "Phone", "phone_id")
        tx.merge(receiver_node, "Phone", "phone_id")
        # Call relationship with properties
        props = {"start_time": str(row.get("start_time","")), "duration": float(row.get("duration_seconds") or 0)}
        call_rel = Relationship(caller_node, "CALLED", receiver_node, **props)
        tx.create(call_rel)
        # optional: link phone to device IMEI
        imei = row.get("imei")
        if pd.notna(imei) and imei != "":
            imei_id = str(imei)
            dev_node = Node("Device", imei=imei_id)
            tx.merge(dev_node, "Device", "imei")
            r2 = Relationship(dev_node, "USED_BY_PHONE", caller_node)
            tx.merge(r2)
    tx.commit()

def ingest_devices_to_neo4j(graph, devices_df):
    """Ingest device -> phone or imei -> phone relationships from a device mapping file."""
    devices_df = devices_df.copy()
    ensure_cols(devices_df, ["imei","phone_number","first_seen","last_seen"])
    tx = graph.begin()
    for _, row in tqdm(devices_df.iterrows(), total=len(devices_df), desc="Ingesting devices to Neo4j"):
        imei = row.get("imei")
        phone = row.get("phone_number")
        if pd.isna(imei):
            continue
        dev_node = Node("Device", imei=str(imei))
        tx.merge(dev_node, "Device", "imei")
        if pd.notna(phone):
            phone_id = anonymize_id(phone)
            phone_node = Node("Phone", phone_id=phone_id, raw_phone=str(phone)[:60])
            tx.merge(phone_node, "Phone", "phone_id")
            r = Relationship(dev_node, "ASSOCIATED_WITH", phone_node)
            tx.merge(r)
    tx.commit()

def ingest_locations_to_neo4j(graph, loc_df):
    """Create Location nodes and link cases / phones / persons by proximity if coordinates present."""
    loc_df = loc_df.copy()
    ensure_cols(loc_df, ["loc_id","latitude","longitude","name","type"])
    tx = graph.begin()
    for _, row in tqdm(loc_df.iterrows(), total=len(loc_df), desc="Ingesting Locations to Neo4j"):
        lid_raw = row.get("loc_id") or f"loc_{_}"
        lid = str(lid_raw)
        lnode = Node("Location", loc_id=lid, name=str(row.get("name",""))[:120], latitude=row.get("latitude"), longitude=row.get("longitude"), loc_type=row.get("type"))
        tx.merge(lnode, "Location", "loc_id")
    tx.commit()

def link_entities_by_location_proximity(graph, search_radius_m=100):
    """
    Create PROXIMITY edges between Location and Case nodes when coordinates are within search_radius_m.
    This uses a coarse approach: pull Case nodes with lat/lon and Location nodes with lat/lon and create edges.
    """
    # We run Cypher directly: this requires spatial functions availability. We'll use naive pairwise approach
    cypher_pull_cases = "MATCH (c:Case) WHERE exists(c.latitude) AND exists(c.longitude) RETURN c.case_id as case_id, toFloat(c.latitude) as lat, toFloat(c.longitude) as lon"
    cypher_pull_locs = "MATCH (l:Location) WHERE exists(l.latitude) AND exists(l.longitude) RETURN l.loc_id as loc_id, toFloat(l.latitude) as lat, toFloat(l.longitude) as lon"
    cases = list(graph.run(cypher_pull_cases).data())
    locs = list(graph.run(cypher_pull_locs).data())
    # convert to simple lists and compute distances (Haversine)
    import math
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000.0
        phi1 = math.radians(lat1); phi2 = math.radians(lat2)
        dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    tx = graph.begin()
    for c in tqdm(cases, desc="Linking cases to locations by proximity"):
        for l in locs:
            d = haversine(float(c["lat"]), float(c["lon"]), float(l["lat"]), float(l["lon"]))
            if d <= search_radius_m:
                # create relationship
                ca = graph.nodes.match("Case", case_id=c["case_id"]).first()
                lo = graph.nodes.match("Location", loc_id=l["loc_id"]).first()
                if ca is not None and lo is not None:
                    rel = Relationship(ca, "NEAR_LOCATION", lo, distance_meters=float(d))
                    tx.merge(rel)
    tx.commit()

# ---------- NetworkX fallback ----------
def build_graph_networkx(fir_df=None, cdr_df=None, devices_df=None, loc_df=None, salt="graph_salt"):
    G = nx.Graph()
    # FIRs -> Case nodes + Persons + CO_OFFEND edges
    if fir_df is not None:
        fir_df = fir_df.copy()
        ensure_cols(fir_df, ["fir_id","date_time","crime_type","latitude","longitude","accused_list","victim_list"])
        for _, r in fir_df.iterrows():
            cid = str(r["fir_id"])
            G.add_node(f"Case::{cid}", label="Case", case_id=cid, crime_type=str(r.get("crime_type","")), date_time=str(r.get("date_time","")))
            accused = r.get("accused_list")
            if pd.notna(accused):
                if isinstance(accused, str):
                    accused_items = [a.strip() for a in accused.split(";") if a.strip()]
                else:
                    accused_items = list(accused)
                for a in accused_items:
                    pid = anonymize_id(a, salt)
                    G.add_node(f"Person::{pid}", label="Person", person_id=pid)
                    G.add_edge(f"Person::{pid}", f"Case::{cid}", relation="CO_OFFEND")
    # CDRs -> Phone nodes + CALLED edges
    if cdr_df is not None:
        cdr_df = cdr_df.copy()
        ensure_cols(cdr_df, ["cdr_id","caller_id","receiver_id","start_time","duration_seconds","tower_lat","tower_lon","imei"])
        for _, r in cdr_df.iterrows():
            caller = r.get("caller_id"); receiver = r.get("receiver_id")
            if pd.isna(caller) or pd.isna(receiver):
                continue
            cpid = anonymize_id(caller, salt); rpid = anonymize_id(receiver, salt)
            G.add_node(f"Phone::{cpid}", label="Phone", phone_id=cpid, raw_phone=str(caller)[:60])
            G.add_node(f"Phone::{rpid}", label="Phone", phone_id=rpid, raw_phone=str(receiver)[:60])
            G.add_edge(f"Phone::{cpid}", f"Phone::{rpid}", relation="CALLED", start_time=str(r.get("start_time","")), duration=float(r.get("duration_seconds") or 0))
            # optional: connect phone to device
            imei = r.get("imei")
            if pd.notna(imei):
                G.add_node(f"Device::{imei}", label="Device", imei=str(imei))
                G.add_edge(f"Device::{imei}", f"Phone::{cpid}", relation="USED_BY_PHONE")
    # Devices mapping
    if devices_df is not None:
        devices_df = devices_df.copy()
        ensure_cols(devices_df, ["imei","phone_number"])
        for _, r in devices_df.iterrows():
            imei = r.get("imei"); phone = r.get("phone_number")
            if pd.notna(imei):
                G.add_node(f"Device::{imei}", label="Device", imei=str(imei))
                if pd.notna(phone):
                    pid = anonymize_id(phone, salt)
                    G.add_node(f"Phone::{pid}", label="Phone", phone_id=pid, raw_phone=str(phone)[:60])
                    G.add_edge(f"Device::{imei}", f"Phone::{pid}", relation="ASSOCIATED_WITH")
    # Locations
    if loc_df is not None:
        loc_df = loc_df.copy()
        ensure_cols(loc_df, ["loc_id","latitude","longitude","name"])
        for _, r in loc_df.iterrows():
            lid = r.get("loc_id") or f"loc_{_}"
            G.add_node(f"Location::{lid}", label="Location", loc_id=str(lid), latitude=r.get("latitude"), longitude=r.get("longitude"), name=str(r.get("name","")))
    return G

# ---------- Top-level orchestration ----------
def build_graph_from_files(fir_path=None, cdr_path=None, device_path=None, loc_path=None, mode="networkx", neo4j_config=None, salt="graph_salt", **kwargs):
    """
    mode: "neo4j" or "networkx"
    neo4j_config: dict with keys uri,user,password (required for neo4j mode)
    Files can be CSV or GeoJSON (we use pandas.read_csv or pandas.read_json/gpd.read_file)
    """
    # load data
    def try_read(path):
        if path is None or not os.path.exists(path):
            return None
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in [".csv", ".txt"]:
                return pd.read_csv(path)
            elif ext in [".json", ".geojson"]:
                import geopandas as gpd
                return gpd.read_file(path)
            elif ext in [".parquet"]:
                return pd.read_parquet(path)
            else:
                # try csv
                return pd.read_csv(path)
        except Exception as e:
            print(f"Failed reading {path}: {e}")
            return None

    fir_df = try_read(fir_path)
    cdr_df = try_read(cdr_path)
    devices_df = try_read(device_path)
    loc_df = try_read(loc_path)

    # normalise some columns for typical expectations
    if fir_df is not None:
        if "accused" in fir_df.columns and "accused_list" not in fir_df.columns:
            fir_df = fir_df.rename(columns={"accused":"accused_list"})
        if "victim" in fir_df.columns and "victim_list" not in fir_df.columns:
            fir_df = fir_df.rename(columns={"victim":"victim_list"})

    if mode == "neo4j":
        if neo4j_config is None:
            raise ValueError("neo4j_config must be provided in neo4j mode")
        graph = connect_neo4j(neo4j_config["uri"], neo4j_config["user"], neo4j_config["password"])
        create_neo4j_schema(graph)
        # ingest nodes & edges
        if fir_df is not None:
            ingest_fir_cases_to_neo4j(graph, fir_df, salt=salt)
        if cdr_df is not None:
            ingest_cdr_to_neo4j(graph, cdr_df, salt=salt)
        if devices_df is not None:
            ingest_devices_to_neo4j(graph, devices_df)
        if loc_df is not None:
            ingest_locations_to_neo4j(graph, loc_df)
        # optionally link by proximity (expensive)
        if kwargs.get("link_by_proximity", False):
            search_radius = kwargs.get("proximity_meters", 100)
            link_entities_by_location_proximity(graph, search_radius_m=search_radius)
        print("Neo4j ingestion finished.")
        return graph

    elif mode == "networkx":
        G = build_graph_networkx(fir_df=fir_df, cdr_df=cdr_df, devices_df=devices_df, loc_df=loc_df, salt=salt)
        # optional GraphML export
        out_graphml = kwargs.get("out_graphml")
        if out_graphml:
            nx.write_graphml(G, out_graphml)
            print(f"Exported GraphML to {out_graphml}")
        return G
    else:
        raise ValueError("mode must be 'neo4j' or 'networkx'")

# If run as script demonstration (not executed in imports)
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fir", help="FIR CSV/GeoJSON path")
    p.add_argument("--cdr", help="CDR CSV path")
    p.add_argument("--devices", help="Device mapping CSV path")
    p.add_argument("--locs", help="Locations CSV/GeoJSON path")
    p.add_argument("--mode", choices=["neo4j","networkx"], default="networkx")
    p.add_argument("--out_graphml", help="Output GraphML path (networkx mode)")
    p.add_argument("--neo4j_uri", help="Neo4j bolt uri")
    p.add_argument("--neo4j_user", help="Neo4j user")
    p.add_argument("--neo4j_pass", help="Neo4j password")
    args = p.parse_args()
    neo_cfg = None
    if args.mode == "neo4j":
        neo_cfg = {"uri":args.neo4j_uri, "user":args.neo4j_user, "password":args.neo4j_pass}
    g = build_graph_from_files(fir_path=args.fir, cdr_path=args.cdr, device_path=args.devices, loc_path=args.locs, mode=args.mode, neo4j_config=neo_cfg, out_graphml=args.out_graphml)
    print("Done.")