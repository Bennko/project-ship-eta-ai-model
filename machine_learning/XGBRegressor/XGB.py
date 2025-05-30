import pandas as pd
import numpy as np
import sys 
from numpy import mean, std
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import randint
import xgboost as xgb

#use given csv data for the model
data = pd.read_csv("../../data/RotHam_cleaned/rotterdam_hamburg_clean.csv", on_bad_lines="warn")
print('Data read done')

#specify test features
test_features = [ "COG", "TH", "shiptype", "EndLongitude", "EndLatitude", "pastTravelTime"]
print('Specify test features done')

#specify test and training sets
#Random state is used for initializing the internal random number generator, which will decide the splitting of data into train and test indices
y = data["timeTillArrival"]
X = data[["Latitude", "Longitude", "SOG"] + test_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Spliting data done')

n_estimators_list = [50, 100, 150]
max_depth_list = [5, 10, 15]

#Loop through the different parameters
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        #choose XGBRegressor as model and train it with the different values
        model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=0.05, max_depth=max_depth, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_alpha=0.1, reg_lambda=0.1, objective='reg:squarederror', random_state=42)
        model.fit(X_train,y_train)
        print(f'Fit model done with n_estimators = {n_estimators} and max_depth = {max_depth}')
        #use fractions of data for prediction
        y_pred = model.predict(X_test)
        print(f'Prediction done with n_estimators = {n_estimators} and max_depth = {max_depth}')

        #Evaluate the model (Perfect MAE = 0)
        #Give out MAE of the prediction set compared to the test set
        #MAE in minutes
        mse = mean_absolute_error(y_test, y_pred)
        print(f'MAE with n_estimators = {n_estimators} and max_depth = {max_depth}: ' , mse/60)
        print("\n")