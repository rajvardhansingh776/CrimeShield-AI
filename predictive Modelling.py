import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
from joblib import dump, load
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm

def make_spatial_grid(gdf_points, cell_size_m=500, crs_epsg=3857):
    g = gdf_points.to_crs(epsg=crs_epsg)
    bounds = g.total_bounds
    minx, miny, maxx, maxy = bounds
    x_coords = np.arange(minx, maxx + cell_size_m, cell_size_m)
    y_coords = np.arange(miny, maxy + cell_size_m, cell_size_m)
    polys = []
    ids = []
    for i, x in enumerate(x_coords[:-1]):
        for j, y in enumerate(y_coords[:-1]):
            polys.append(box(x, y, x + cell_size_m, y + cell_size_m))
            ids.append(f"cell_{i}_{j}")
    grid = gpd.GeoDataFrame({"cell_id": ids, "geometry": polys}, crs=f"EPSG:{crs_epsg}")
    grid = grid.to_crs(epsg=4326)
    return grid

def assign_points_to_grid(incidents_gdf, grid_gdf):
    pts = incidents_gdf.copy()
    pts = pts.to_crs(epsg=4326)
    grid = grid_gdf.to_crs(epsg=4326)
    joined = gpd.sjoin(pts, grid, how="left", predicate="within")
    return joined

def aggregate_by_timecell(joined_points, time_col="date_time", freq="24H"):
    df = joined_points.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["time_bin"] = df[time_col].dt.floor(freq)
    agg = df.groupby(["cell_id", "time_bin"]).size().reset_index(name="crime_count")
    return agg

def build_feature_matrix(agg, grid, history_windows=[1,3,7,14], freq="24H", demog_df=None):
    agg2 = agg.copy()
    agg2 = agg2.set_index(["cell_id","time_bin"]).sort_index()
    unique_cells = agg2.index.get_level_values(0).unique()
    rows = []
    for (cell, time) in tqdm(agg2.index):
        row = {"cell_id": cell, "time_bin": time}
        for w in history_windows:
            start = time - pd.Timedelta(days=w)
            mask = (agg2.index.get_level_values(0) == cell) & (agg2.index.get_level_values(1) > start) & (agg2.index.get_level_values(1) <= time)
            try:
                s = agg2.loc[cell].loc[start + pd.Timedelta(hours=0):time]["crime_count"].sum()
            except Exception:
                s = 0
            row[f"hist_{w}d"] = s
        row["hour"] = pd.to_datetime(time).hour
        row["weekday"] = pd.to_datetime(time).weekday()
        rows.append(row)
    feat = pd.DataFrame(rows)
    if demog_df is not None:
        feat = feat.merge(demog_df[["cell_id"] + [c for c in demog_df.columns if c!="geometry" and c!="cell_id"]], on="cell_id", how="left")
    for c in feat.columns:
        if feat[c].dtype == "object":
            feat[c] = feat[c].fillna("unk")
    feat = feat.fillna(0)
    return feat

def make_labels(agg, lead_hours=24, freq="24H"):
    agg2 = agg.copy()
    agg2 = agg2.set_index(["cell_id","time_bin"]).sort_index()
    rows = []
    for (cell, time), group in agg2.groupby(level=0):
        try:
            future_bin = time + pd.Timedelta(hours=lead_hours)
            future_count = agg2.loc[(cell, future_bin)]["crime_count"]
            label = 1 if future_count > 0 else 0
        except Exception:
            label = 0
        rows.append({"cell_id": cell, "time_bin": time, "label": label})
    lab = pd.DataFrame(rows)
    return lab

def train_random_forest(X, y, n_estimators=200, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=random_state)
    clf.fit(X_train_s, y_train)
    preds = clf.predict_proba(X_test_s)[:,1]
    auc = roc_auc_score(y_test, preds) if len(np.unique(y_test))>1 else None
    return {"model": clf, "scaler": scaler, "auc": auc}

def prepare_lstm_sequences(agg, grid_cells, seq_len=7, freq="24H"):
    pivot = agg.pivot(index="time_bin", columns="cell_id", values="crime_count").fillna(0).sort_index()
    seqs = []
    targets = []
    cell_list = list(grid_cells["cell_id"])
    times = pivot.index
    for i in range(len(times)-seq_len):
        seq = pivot.iloc[i:i+seq_len][cell_list].values
        target = pivot.iloc[i+seq_len][cell_list].values
        seqs.append(seq)
        targets.append(target)
    X = np.array(seqs)
    y = np.array(targets)
    return X, y, cell_list, pivot.index

def build_lstm_model(n_cells, seq_len, hidden=64):
    model = models.Sequential()
    model.add(layers.Input(shape=(seq_len, n_cells)))
    model.add(layers.LSTM(hidden, return_sequences=False))
    model.add(layers.Dense(n_cells, activation="relu"))
    model.compile(optimizer="adam", loss="mse")
    return model

def train_lstm(X, y, epochs=10, batch_size=16):
    scaler = MinMaxScaler()
    nsamples, seq_len, n_cells = X.shape
    X_flat = X.reshape(nsamples*seq_len, n_cells)
    X_flat_s = scaler.fit_transform(X_flat)
    X_s = X_flat_s.reshape(nsamples, seq_len, n_cells)
    y_flat = y.reshape(nsamples, n_cells)
    model = build_lstm_model(n_cells, seq_len, hidden=64)
    model.fit(X_s, y_flat, epochs=epochs, batch_size=batch_size, verbose=1)
    return {"model": model, "scaler": scaler}

def compute_kde_for_recent(incidents_gdf, bandwidth=0.001, cells_gdf=None, last_hours=72):
    pts = incidents_gdf.copy()
    pts["date_time"] = pd.to_datetime(pts["date_time"])
    cutoff = pts["date_time"].max() - pd.Timedelta(hours=last_hours)
    recent = pts[pts["date_time"] >= cutoff]
    if recent.empty:
        cells_gdf["kde_score"] = 0.0
        return cells_gdf
    recent_proj = recent.to_crs(epsg=3857)
    coords = np.vstack([recent_proj.geometry.x.values, recent_proj.geometry.y.values]).T
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(coords)
    grid_proj = cells_gdf.to_crs(epsg=3857)
    centroid_coords = np.vstack([grid_proj.geometry.centroid.x.values, grid_proj.geometry.centroid.y.values]).T
    log_density = kde.score_samples(centroid_coords)
    cells_gdf["kde_score"] = np.exp(log_density)
    cells_gdf["kde_score"] = (cells_gdf["kde_score"] - cells_gdf["kde_score"].min()) / (cells_gdf["kde_score"].max() - cells_gdf["kde_score"].min() + 1e-9)
    return cells_gdf

def predict_hotspots(fir_gdf, cdr_gdf, demog_gdf=None, cell_size_m=500, seq_len_days=7):
    incidents = fir_gdf.copy()
    incidents = incidents.dropna(subset=["geometry"])
    grid = make_spatial_grid(incidents, cell_size_m=cell_size_m)
    joined = assign_points_to_grid(incidents, grid)
    agg = aggregate_by_timecell(joined, time_col="date_time", freq="24H")
    feat = build_feature_matrix(agg, grid, history_windows=[1,3,7,14], freq="24H", demog_df=demog_gdf)
    lab = make_labels(agg, lead_hours=24, freq="24H")
    df_merge = feat.merge(lab, on=["cell_id","time_bin"], how="left").fillna(0)
    X = df_merge[[c for c in df_merge.columns if c not in ["cell_id","time_bin","label"]]]
    y = df_merge["label"].astype(int)
    rf_info = train_random_forest(X, y)
    cells = grid.copy()
    latest_time = agg["time_bin"].max()
    last_row = df_merge[df_merge["time_bin"]==latest_time]
    if last_row.empty:
        X_latest = X.iloc[-1:,:]
    else:
        X_latest = last_row[[c for c in X.columns]]
    X_latest_s = rf_info["scaler"].transform(X_latest)
    rf_probs_24h = rf_info["model"].predict_proba(X_latest_s)[:,1]
    cells["rf_prob_24h"] = rf_probs_24h
    X_week = X_latest.copy()
    cells["rf_prob_7d"] = cells["rf_prob_24h"]
    X_lstm, y_lstm, cell_list, pivot_times = prepare_lstm_sequences(agg, grid, seq_len=seq_len_days, freq="24H")
    if X_lstm.shape[0] > 0:
        lstm_info = train_lstm(X_lstm, y_lstm, epochs=6, batch_size=8)
        last_seq = X_lstm[-1:]
        ns, sl, nc = last_seq.shape
        last_seq_flat = last_seq.reshape(ns*sl, nc)
        last_seq_s = lstm_info["scaler"].transform(last_seq_flat).reshape(ns, sl, nc)
        pred = lstm_info["model"].predict(last_seq_s)
        pred_counts = pred.flatten()
        cell_index = [c for c in cell_list]
        pred_df = pd.DataFrame({"cell_id": cell_index, "lstm_pred_next24_count": pred_counts})
        cells = cells.merge(pred_df[["cell_id","lstm_pred_next24_count"]], on="cell_id", how="left")
        cells["lstm_prob_24h"] = (cells["lstm_pred_next24_count"]>0).astype(float)
    else:
        cells["lstm_prob_24h"] = 0.0
    kde_cells = compute_kde_for_recent(incidents, bandwidth=1000, cells_gdf=cells, last_hours=72)
    cells["kde_score"] = kde_cells["kde_score"].values
    cells["combined_prob_24h"] = (0.5*cells["rf_prob_24h"] + 0.3*cells["lstm_prob_24h"] + 0.2*cells["kde_score"])
    cells["combined_prob_24h"] = (cells["combined_prob_24h"] - cells["combined_prob_24h"].min()) / (cells["combined_prob_24h"].max() - cells["combined_prob_24h"].min() + 1e-9)
    cells["combined_prob_7d"] = cells["combined_prob_24h"]
    return {"grid": cells, "rf_info": rf_info}