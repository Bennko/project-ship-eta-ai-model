import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


df = pd.read_csv('../../data/RotHam_cleaned/rotterdam_hamburg_clean.csv', low_memory=False)

features = ['Latitude', 'Longitude', 'SOG', 'COG','TH']
training_features = df[features]
df['EndTime'] = pd.to_datetime(df['EndTime'], errors='raise')
df['StartTime'] = pd.to_datetime(df['StartTime'], errors = 'raise')
df['time'] = pd.to_datetime(df['time'], errors = 'raise')
df['duration_hours'] = (df['EndTime'] - df['time']).dt.total_seconds() / 3600
target = df['duration_hours']

features_train, features_test, target_train, target_test = train_test_split(training_features, target, test_size = 0.3, train_size = 0.7, random_state = 42)

knnRegressor = KNeighborsRegressor(n_neighbors=5)
knnRegressor.fit(features_train, target_train)
y_predict = knnRegressor.predict(features_test)

plt.figure(figsize=(10, 6))
plt.scatter(target_test, y_predict, alpha=0.6)
plt.plot([target_test.min(), target_test.max()], [target_test.min(), target_test.max()], 'k--', lw=2)
plt.xlabel('Actual Arrival Time (hours)')
plt.ylabel('Predicted Arrival Time (hours)')
plt.title('Actual vs Predicted Arrival Times')
plt.show()