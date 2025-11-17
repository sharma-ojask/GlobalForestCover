import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset


def build_features(series_list, pixel_indices, lag, neighbor_map, horizon):
    X_all, y_all = [], []

    for i in pixel_indices:
        series = np.array(series_list[i])
        n = len(series)

        for t in range(lag, n):
            own_lags = series[t-lag:t].tolist()
            neighbor_lags = []
            if neighbor_map is not None:
                neighbors = neighbor_map[i]
                for l in range(1, lag+1):
                    vals = [series_list[n_idx][t-l-horizon] for n_idx in neighbors]
                    neighbor_lags.append(np.mean(vals))
            X_all.append(own_lags + neighbor_lags)
            y_all.append(series[t])

    return np.array(X_all), np.array(y_all)

def train_xgb(series_list, coords, k, horizon):
    # Needed inputs: 
    # List of list of timeseries values
    # coordinates mtrix nx2, should be portable from spreadsheet
    # number of neighbors (4) 

    coords = np.array(coords)  # shape = (num_pixels, 2), x and y position of pixel
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    neighbor_map = {i: indices[i, 1:] for i in range(len(coords))}  # skip self
    train_pixels, test_pixels = train_test_split(range(len(series_list)), train_size=0.75, random_state=999)
    
    X_train, y_train = build_features(series_list, train_pixels, lag=3, neighbor_map=neighbor_map, horizon = horizon)
    X_test,  y_test  = build_features(series_list, test_pixels,  lag=3, neighbor_map=neighbor_map, horizon = horizon)

    # Train XGB
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=1234)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    #Append predictions to series list
    n = 0
    for i in test_pixels:
        series_list[i].append(y_pred[n])
        n+=1

    #Step through and add features with lags
    for step in range(horizon-1):
        X_test,  y_test  = build_features(series_list, test_pixels,  lag=3, neighbor_map=neighbor_map, horizon = horizon)
        y_pred = model.predict(X_test)
        n = 0
        for i in test_pixels:
            series_list[i].append(y_pred[n])
            n+=1


    return series_list

def XGB_horizon(filepath, horizon):
    data = pd.read_csv(filepath)
    data = data[data['2000'] >20]
    coords = data[['x', 'y']].copy()
    data = data.drop(columns=["x_bin","y_bin","x","y", "id"])
    series_list = []
    
    for _,row in data.iterrows():
        series_list.append(row.values.flatten().tolist())
    old_data = []
    new_data = []
    for series in series_list:
        old_data.append(series[:-horizon])  
        new_data.append(series[-horizon:])  
    
    series_list_with_preds = train_xgb(old_data,coords,4, horizon= horizon)
    
    preds = []
    actuals = []
    for i in range(len(series_list_with_preds)):
        if len(series_list_with_preds[i]) ==25:
            preds.append(series_list_with_preds[i])
            actuals.append(series_list[i])
    
    return actuals ,preds


data = pd.read_csv("data/course_data.csv")
data = data[1:600]
coords = data[['x', 'y']].copy()
data_copy = data.copy()
data = data.drop(columns=["x_bin","y_bin","x","y", "id"])
series_list = []
series_list_2 = []

for _,row in data.iterrows():
    series_list.append(row.values.flatten().tolist())
for _,row in data_copy.iterrows():
    series_list_2.append(row.values.flatten().tolist())

scaler = TimeSeriesScalerMeanVariance()
dataset_scaled = scaler.fit_transform(series_list)

model = TimeSeriesKMeans(n_clusters=5, metric="dtw", random_state=0)
labels = model.fit_predict(dataset_scaled)

dataset_new = []

for i in range(len(labels)):
    if labels[i] == 0:
        dataset_new.append(series_list_2[i])
df = pd.DataFrame(dataset_new, columns =data_copy.columns)

print(df)
df.to_csv("results_cluster0.csv", index = False)

preds, actual = XGB_horizon("results_cluster0.csv",7)
print(preds)
