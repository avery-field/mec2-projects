import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#load data
yellow_tripdata = pd.read_parquet('/Users/averyfield/Downloads/yellow_tripdata_2022-01.parquet', engine='pyarrow')
print(yellow_tripdata.head())

#preprocessing
yellow_tripdata = yellow_tripdata.dropna()

yellow_tripdata['trip_duration_minutes'] = yellow_tripdata['tpep_dropoff_datetime'] - yellow_tripdata['tpep_pickup_datetime']
yellow_tripdata['trip_duration_minutes'] = yellow_tripdata['trip_duration_minutes'].dt.total_seconds().div(60).astype(int)
#print(yellow_tripdata['trip_duration_minutes'])

feature_col = list(yellow_tripdata.columns)
#print(feature_col)

X = yellow_tripdata.loc[:, yellow_tripdata.columns != 'total_amount']
y = yellow_tripdata['total_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#baseline model
from sklearn.dummy import DummyRegressor
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)

print('Baseline MAE:')
print(mean_absolute_error(y_test, y_pred))

#more preprocessing
from sklearn.preprocessing import FunctionTransformer

#separate categorical and continuous data
categorical_columns = ['VendorID', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'payment_type']
continuous_columns = ['passenger_count', 'trip_distance', 'fare_amount',
                      'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 
                      'congestion_surcharge', 'airport_fee', 'trip_duration_minutes']
datetime_columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime',]

#preprocessing
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

continuous_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

datetime_transformer = Pipeline(steps=[
    ('convert_to_unix', FunctionTransformer(lambda x: x.astype(int) // 10**9))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_columns),
        ('cont', continuous_transformer, continuous_columns),
        ('datetime', datetime_transformer, datetime_columns)
    ])

#Linear Regression Training
linear_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

linear_regression.fit(X_train, y_train)

#Linear Regression Analysis
linreg_pred = linear_regression.predict(X_test)

print(X_train.head(10))

print('Linear Regression Predictions:')
print(linreg_pred)

print('Linear Regression MEA:')
print(mean_absolute_error(y_test, linreg_pred))

#Random Forest Regression Training
rf_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=5, random_state=42, n_jobs=-1) )
])

rf_regression.fit(X_train, y_train)

#Random Forest Regression Analysis
rf_pred = rf_regression.predict(X_test)

print('Random Forest Regression Predictions:')
print(rf_pred)

print('Random Forest MEA:')
print(mean_absolute_error(y_test, rf_pred))

print('RF score:')
print(rf_regression.score(X_test, y_test))

#Grid Search CV (Where I'm getting errors)
param_grid = {
    'regressor__n_estimators': [5, 10, 15],
    'regressor__max_depth': [10, 20, 30],
    'regressor__min_samples_split': [8, 10, 12]

}

grid_search = GridSearchCV(estimator = rf_regression, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2, scoring='neg_mean_absolute_error')


grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

