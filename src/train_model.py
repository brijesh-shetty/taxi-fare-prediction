# ============================================
# Taxi Trip Duration Prediction
# Models: Linear, Tree, RF, XGBoost, ExtraTrees, MLP
# ============================================

import random
import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# ---------------- CONFIG ----------------
FILE = "data/combined_ML_data.csv"   # dataset path
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------------------------------
# 1. LOAD DATA
# ---------------------------------------
df = pd.read_csv(FILE)
print("Loaded dataset shape:", df.shape)

# Convert datetime columns
df['tpep_pickup_datetime'] = pd.to_datetime(
    df['tpep_pickup_datetime'], errors='coerce'
)
df['tpep_dropoff_datetime'] = pd.to_datetime(
    df['tpep_dropoff_datetime'], errors='coerce'
)

# Compute trip duration (minutes)
df['trip_duration'] = (
    df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
).dt.total_seconds() / 60

# ---------------------------------------
# 2. DATA CLEANING
# ---------------------------------------
df = df[df['trip_distance'] > 0]
df = df[df['fare_amount'] > 0]
df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 180)]

print("After cleaning:", df.shape)

# ---------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday

# Cyclic encoding
df['pickup_hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
df['pickup_hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)

# Derived feature
df['tip_rate'] = df['tip_amount'] / (df['fare_amount'] + 1e-6)

# Drop non-numeric columns
df = df.drop(columns=[
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime'
], errors='ignore')

# ---------------------------------------
# 4. DEFINE X AND y
# ---------------------------------------
y = df['trip_duration'].values

X = df.drop(columns=[
    'trip_duration',
    'congestion_surcharge',
    'mta_tax',
    'payment_type',
    'pickup_hour',
    'improvement_surcharge',
    'VendorID'
], errors='ignore')

print("Final feature shape:", X.shape)

# ---------------------------------------
# 5. SCALING
# ---------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------
# 6. TRAIN-TEST SPLIT
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------------
# 7. EVALUATION FUNCTION
# ---------------------------------------
def evaluate(model, name):
    pred = model.predict(X_test)
    print(f"\n{name}")
    print("MAE :", mean_absolute_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("R2  :", r2_score(y_test, pred))

# ---------------------------------------
# 8. MODELS
# ---------------------------------------

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate(lr, "Linear Regression")

# Decision Tree
dt = DecisionTreeRegressor(max_depth=14, random_state=42)
dt.fit(X_train, y_train)
evaluate(dt, "Decision Tree")

# Random Forest
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42
)
rf.fit(X_train, y_train)
evaluate(rf, "Random Forest")

# XGBoost
xgb = XGBRegressor(
    n_estimators=400,
    learning_rate=0.07,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42
)
xgb.fit(X_train, y_train)
evaluate(xgb, "XGBoost")

# Extra Trees
etr = ExtraTreesRegressor(
    n_estimators=400,
    random_state=42
)
etr.fit(X_train, y_train)
evaluate(etr, "Extra Trees")

# MLP Regressor
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),
    max_iter=300,
    random_state=42
)
mlp.fit(X_train, y_train)
evaluate(mlp, "MLP Regressor")
