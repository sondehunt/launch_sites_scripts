import csv
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point
from matplotlib import pyplot as plt


def ckdnearest(data_frame_a: gpd.GeoDataFrame, data_frame_b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    nA = np.array(list(zip(data_frame_a.geometry.x, data_frame_a.geometry.y)))
    nB = np.array(list(zip(data_frame_b.geometry.x, data_frame_b.geometry.y)))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdf = pd.concat(
        [data_frame_a, data_frame_b.loc[idx, data_frame_b.columns != 'geometry'].reset_index(),
         pd.Series(dist, name='dist')], axis=1)
    return gdf


print('Reading sonde models...')
wmo_to_model_index = pd.DataFrame(data={})
models = pd.DataFrame(data={})
with open('sonde_models.csv') as models_file:
    models_csv = csv.DictReader(models_file, delimiter=",")
    i = 0
    for row in models_csv:
        wmo_ids = row['wmo type identifier'].split(',')
        for id in wmo_ids:
            real_wmo = id
            autosonde = False
            if str(id).startswith("1"):
                real_wmo = str(id)[1:]
            if real_wmo.endswith("(autosonde)"):
                real_wmo = real_wmo[:real_wmo.find("(autosonde)")]
                autosonde = True
            wmo_to_model_index = wmo_to_model_index.append(pd.DataFrame(data={
                'WMO': [real_wmo],
                'ModelIndex': [i],
                'Autosonde': [autosonde],
            }), ignore_index=True)
        models = models.append(pd.DataFrame(data={
            'Manufacturer': row['Manufacturer'],
            'Family': row['Family'],
            'Model': row['Model'],
            'Frequency': row['Frequency'],
            'XDATA': None if row['XDATA/ext.Sensor capable'] == '?' else bool(row['XDATA/ext.Sensor capable']),
            'Windfinding': None if row['internal windfinding'] == '?' else bool(row['internal windfinding']),
            'Autosonde': None if row['autosonde capable'] == '?' else bool(row['autosonde capable']),
            'WMOTypeIdentifier': row['wmo type identifier'],
            'WMOTypeIdentifierFamily': row['wmo type identifier (family)'],
        }, index=[i]))
        i += 1

print('Reading stations list (with WMO ID)...')
stations = gpd.GeoDataFrame(data={})
with open('stations_2019jun3.txt') as stations_file:
    i = 0
    for line in stations_file:
        wmo_id, rest = line.strip().split(':')
        sonde_models, rest = rest.strip().split('#')
        lat, long, *rest = rest.strip().split(' ')
        stations = stations.append(gpd.GeoDataFrame(data={
            'WMO': int(wmo_id),
            'SondeModels': [sonde_models],
            'ApproxLatitude': float(lat),
            'ApproxLongitude': float(long),
        }, index=[i]))
        i += 1
stations.geometry = gpd.points_from_xy(stations.ApproxLongitude, stations.ApproxLatitude)

print('Reading station search results...')
station_results = gpd.GeoDataFrame(data={})
with open('StationSearchResults.csv') as results_file:
    station_csv = csv.DictReader(results_file, delimiter=";")
    i = 0
    for row in station_csv:
        lat, long = float(row['Latitude']), float(row['Longitude'])
        entry = gpd.GeoDataFrame(data=row, index=[i])
        entry.Latitude = [lat]
        entry.Longitude = [long]
        station_results = station_results.append(entry)
        i += 1
station_results.geometry = gpd.points_from_xy(station_results.Longitude, station_results.Latitude)

print('Matching nearest stations....')
nearest = ckdnearest(stations, station_results)


def sonde_df_to_dict(df: pd.DataFrame, match_df: pd.DataFrame) -> dict:
    dicts = []
    i = 0
    for index, row in df.iterrows():
        dicts.append({
            'model': {
                'manufacturer': row['Manufacturer'],
                'family': row['Family'],
                'model': row['Model'],
                'wmo': row['WMOTypeIdentifier'],
            },
            'autosonde': bool(match_df.iloc[i].Autosonde),
        })
        i += 1
    return dicts


station_jsons = []
for index, row in nearest.iterrows():
    sonde_models = row['SondeModels'].strip().split(',')
    sonde_models_matches = wmo_to_model_index[wmo_to_model_index['WMO'].isin(sonde_models)]
    sonde_models_indices = sonde_models_matches['ModelIndex']
    sonde_models_df = models.loc[sonde_models_indices]
    station = {
        'name': row['Station'],
        'operator': row['Supervising organization'],
        'wmo-id': row['WMO'],
        'latitude': row['Latitude'],
        'longitude': row['Longitude'],
        'elevation': row['Elevation'],
        'sonde_wmos': sonde_models,
        'sonde': sonde_df_to_dict(sonde_models_df, sonde_models_matches),
    }
    station_jsons.append(station)

print('Writing stations to stations.json...')
station_json_file = open('stations.json', 'w')
station_json_file.write(json.dumps(station_jsons, indent=4))
station_json_file.close()

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
base = world.plot(color='white', edgecolor='black')
nearest.plot(ax=base, marker='o', color='red', markersize=5)
plt.show()
