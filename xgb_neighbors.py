import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/course_data.csv")
df_original = df.copy()

#Use through 2015 for training

df_firstyears = df.iloc[:, :20] 
df_id = df[["id"]]       
df = pd.concat([df_id, df_firstyears], axis=1)
df.columns = df.columns.str.strip()

#Prevent double naming of y
df.rename(columns={'x':'x_loc','y':'y_loc'}, inplace=True)

# Identify year columns
year_cols = [col for col in df.columns if col.isdigit()]

# Convert to ID, Year, Value
df_long = df.melt(
    id_vars=['id','x_loc','y_loc'],
    value_vars=year_cols,
    var_name='year',
    value_name='value'
)
#Rename "value" to y
df_long.rename(columns={'id':'unique_id','value':'y'}, inplace=True)
df_long['year'] = df_long['year'].astype(int)

max_lag = 3 #How far into the past to look
df_long = df_long.sort_values(['unique_id','year'])
for lag in range(1, max_lag+1):
    df_long[f'y_lag{lag}'] = df_long.groupby('unique_id')['y'].shift(lag)


k = 4  # number of neighbors
coords = df_long[['unique_id','x_loc','y_loc']].drop_duplicates().reset_index(drop=True)
nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords[['x_loc','y_loc']])
distances, indices = nbrs.kneighbors(coords[['x_loc','y_loc']])

# Remove self (first column)
neighbor_ids = indices[:,1:]

# Map unique_id -> neighbor_ids
neighbor_map = dict(zip(coords['unique_id'], neighbor_ids))

# Initialize neighbor lag columns
for lag in range(1, max_lag+1):
    df_long[f'neighbor_y_lag{lag}'] = np.nan

# Fill neighbor lag features
for uid in df_long['unique_id'].unique():
    neighbors = coords.loc[neighbor_map[uid], 'unique_id'].values
    mask = df_long['unique_id'] == uid
    for lag in range(1, max_lag+1):
        # mean of neighbors' lagged y
        neighbor_lag_col = f'y_lag{lag}'
        neighbor_mean = df_long[df_long['unique_id'].isin(neighbors)][['year', neighbor_lag_col]].groupby('year').mean()
        df_long.loc[mask, f'neighbor_y_lag{lag}'] = df_long.loc[mask, 'year'].map(neighbor_mean[neighbor_lag_col])

# Drop nas
df_long = df_long.dropna().reset_index(drop=True)


features = ['x_loc','y_loc','year'] + [f'y_lag{lag}' for lag in range(1,max_lag+1)] + [f'neighbor_y_lag{lag}' for lag in range(1,max_lag+1)]
target = 'y'

X = df_long[features]
y = df_long[target]

print(X)
print(y)

model = XGBRegressor(n_estimators=500,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=567)
model.fit(X, y)


#How far to predict
horizon = 9
last_known = df_long.groupby('unique_id').tail(1).reset_index(drop=True)

preds = []
for year in range(df_long['year'].max()+1, df_long['year'].max()+1+horizon):
    X_future = last_known.copy()
    X_future['year'] = year
    
    # Update self lag features
    for lag in range(1, max_lag+1):
        X_future[f'y_lag{lag}'] = last_known.groupby('unique_id')['y'].shift(lag).fillna(last_known['y'])
    
    # find average neighbor lag for past n years, where n is max lag
    for lag in range(1, max_lag+1):
        neighbor_lag_col = f'y_lag{lag}'
        neighbor_mean = []
        for uid in X_future['unique_id']:
            neighbors = coords.loc[neighbor_map[uid], 'unique_id'].values
            neighbor_vals = last_known[last_known['unique_id'].isin(neighbors)][neighbor_lag_col].values
            neighbor_mean.append(np.mean(neighbor_vals))
        X_future[f'neighbor_y_lag{lag}'] = neighbor_mean
    
    # Predict next number of years in horizon
    y_future = model.predict(X_future[features])
    X_future['y'] = y_future
    preds.append(X_future[['unique_id','year','y']])
    
    # Update last_known
    last_known = X_future.copy()

preds_df = pd.concat(preds)
preds_df.to_csv("preds.csv")


#Load in original data for comparison
df_full = pd.read_csv("data/course_data.csv")
df_full.columns = df_full.columns.str.strip()
df_full.rename(columns={'x': 'x_loc', 'y': 'y_loc'}, inplace=True)

# Identify year columns again
year_cols_full = [col for col in df_full.columns if col.isdigit()]

# Original dataset loaded back in for comparison
df_long_full = df_full.melt(
    id_vars=['id', 'x_loc', 'y_loc'],
    value_vars=year_cols_full,
    var_name='year',
    value_name='y_actual'
)
df_long_full['year'] = df_long_full['year'].astype(int)

# Filter to predict 2016 and beyond
df_actual = df_long_full[df_long_full['year'] >= 2016][['id', 'year', 'y_actual']]

# Match column name to preds
df_actual.rename(columns={'id': 'unique_id'}, inplace=True)
df_actual.to_csv("actual.csv")
df_results = pd.merge(preds_df, df_actual, on=['unique_id', 'year'], how='inner')


mae = mean_absolute_error(df_results['y'], df_results['y_actual'])
print("MAE:", mae)

df_results.to_csv("results5.csv")