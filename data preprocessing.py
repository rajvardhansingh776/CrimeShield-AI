import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import hashlib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from datetime import timedelta

def anonymize_series(series, salt='salt123'):
    def h(x):
        if pd.isna(x):
            return np.nan
        return hashlib.sha256((str(x)+salt).encode()).hexdigest()[:16]
    return series.map(h)

def load_fir(path, datetime_col='date_time'):
    df = pd.read_csv(path)
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    return df

def clean_fir(df,
              lat_col='latitude',
              lon_col='longitude',
              datetime_col='date_time',
              crime_col='crime_type'):
    df = df.copy()
    df[crime_col] = df[crime_col].str.lower().str.strip()
    df = df.drop_duplicates(subset=['fir_id'])
    df = df[df[datetime_col].notna()]
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    geometry = [Point(xy) if not (pd.isna(xy[0]) or pd.isna(xy[1])) else None for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf

def load_cdr(path, datetime_col='start_time'):
    df = pd.read_csv(path)
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    return df

def clean_cdr(df, tower_lat='tower_lat', tower_lon='tower_lon', datetime_col='start_time'):
    df = df.copy()
    df = df[df[datetime_col].notna()]
    df[tower_lat] = pd.to_numeric(df[tower_lat], errors='coerce')
    df[tower_lon] = pd.to_numeric(df[tower_lon], errors='coerce')
    geometry = [Point(xy) if not (pd.isna(xy[0]) or pd.isna(xy[1])) else None for xy in zip(df[tower_lon], df[tower_lat])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf

def load_demographics(path, geom_col=None):
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        return gdf
    except Exception:
        df = pd.read_csv(path)
        if geom_col and geom_col in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df[geom_col]), crs='EPSG:4326')
            return gdf
        raise

def spatial_join_fir_demog(fir_gdf, demog_gdf):
    fir = fir_gdf.copy()
    dem = demog_gdf.to_crs(epsg=4326)
    joined = gpd.sjoin(fir, dem, how='left', predicate='within')
    return joined

def map_cdr_to_fir(cdr_gdf, fir_gdf, time_window_minutes=60, max_distance_meters=500):
    cdr = cdr_gdf.to_crs(epsg=3857)
    fir = fir_gdf.to_crs(epsg=3857)
    cdr['key'] = 1
    fir['key'] = 1
    cdr_expanded = cdr[['cdr_id','caller_id','receiver_id','start_time','duration_seconds','geometry','key']].copy()
    fir_expanded = fir[['fir_id','date_time','geometry','key']].copy()
    merged = cdr_expanded.merge(fir_expanded, on='key', suffixes=('_cdr','_fir'))
    merged['time_diff_minutes'] = (merged['start_time'] - merged['date_time']).abs().dt.total_seconds()/60.0
    merged = merged[merged['time_diff_minutes'] <= time_window_minutes]
    merged_cdr_geom = gpd.GeoSeries(merged['geometry_cdr'], crs='EPSG:3857')
    merged_fir_geom = gpd.GeoSeries(merged['geometry_fir'], crs='EPSG:3857')
    merged['distance_m'] = merged_cdr_geom.distance(merged_fir_geom)
    merged = merged[merged['distance_m'] <= max_distance_meters]
    merged_small = merged[['fir_id','cdr_id','caller_id','receiver_id','start_time','duration_seconds','time_diff_minutes','distance_m']]
    agg = merged_small.groupby('fir_id').agg({
        'cdr_id': lambda x: list(x),
        'caller_id': lambda x: list(x),
        'receiver_id': lambda x: list(x),
        'start_time': ['min','max','count'],
        'duration_seconds': 'mean',
        'time_diff_minutes': 'mean',
        'distance_m': 'mean'
    })
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    result = fir_gdf.merge(agg, on='fir_id', how='left')
    return result

def feature_engineer(final_gdf, crime_col='crime_type'):
    df = final_gdf.copy()
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    numeric_cols = ['duration_seconds_mean','time_diff_minutes_mean','distance_m_mean','hour','dayofweek']
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    scaler = StandardScaler()
    df[[f+s for s in []]] = df[[c for c in numeric_cols]]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    cat_cols = [crime_col,'victim_gender','suspect_gender']
    for c in cat_cols:
        if c not in df.columns:
            df[c] = 'unknown'
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    cat_arr = ohe.fit_transform(df[cat_cols].astype(str))
    cat_df = pd.DataFrame(cat_arr, columns=[f"cat_{i}" for i in range(cat_arr.shape[1])], index=df.index)
    out = pd.concat([df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
    return out

def anonymize_pii(gdf, pii_cols=['victim_age','victim_gender','suspect_age','suspect_gender','caller_id','receiver_id'], salt='salt123'):
    g = gdf.copy()
    for col in pii_cols:
        if col in g.columns:
            if g[col].dtype == object or g[col].dtype == 'int64':
                g[col] = anonymize_series(g[col].astype(str), salt=salt)
            else:
                g[col] = g[col]
    return g

def save_output(gdf, path_csv='processed_dataset.csv', include_geometry=False):
    df = gdf.copy()
    if not include_geometry and 'geometry' in df.columns:
        df = df.drop(columns=['geometry'])
    df.to_csv(path_csv, index=False)
    return path_csv

# Example pipeline function tying everything together
def run_pipeline(fir_path, cdr_path, demog_path, output_path='processed_dataset.csv'):
    fir = load_fir(fir_path)
    fir_clean = clean_fir(fir)
    cdr = load_cdr(cdr_path)
    cdr_clean = clean_cdr(cdr)
    demog = load_demographics(demog_path)
    fir_dem = spatial_join_fir_demog(fir_clean, demog)
    fir_cdr_mapped = map_cdr_to_fir(cdr_clean, fir_dem)
    anonymized = anonymize_pii(fir_cdr_mapped, pii_cols=['caller_id','receiver_id','victim_age','victim_gender','suspect_age','suspect_gender'])
    fe = feature_engineer(anonymized)
    saved = save_output(fe, path_csv=output_path, include_geometry=False)
    return saved
