import pandas as pd
import numpy as np
import sys 
from numpy import mean, std
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint


#use given csv data for the model
data = pd.read_csv("../../data/RotHam_cleaned/rotterdam_hamburg_clean.csv", on_bad_lines="warn")
print('Data read done')

#insert time, longitude and latitude traveled 
data.insert(0, "pastTravelTime",(pd.to_datetime(data["time"]) - pd.to_datetime(data["StartTime"])).dt.total_seconds())
data.insert(0, "longitudeDistance", data["EndLongitude"] - data["StartLongitude"])
data.insert(0, "latitudeDistance", data["EndLatitude"] - data["StartLatitude"])
print('Data modifying done')

#specify test features
test_features = [ "COG", "TH", "shiptype", "MMSI", "TripID", "EndLongitude", "EndLatitude", "pastTravelTime", "longitudeDistance", "latitudeDistance"]
print('Specify test features done')

#specify test and training sets
#Random state is used for initializing the internal random number generator, which will decide the splitting of data into train and test indices
y = (pd.to_datetime(data["EndTime"]) - pd.to_datetime(data["time"])).dt.total_seconds()
X = data[["Latitude", "Longitude", "SOG"] + test_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Spliting data done')

#choose Extra Trees as model and train it with the train split sets
model = ExtraTreesRegressor(n_estimators=350, min_samples_split=2, min_samples_leaf=1, max_depth=None, random_state=42)

model.fit(X_train,y_train)
print('Fit model done')
#use fractions of data for prediction
y_pred = model.predict(X_test)
print('Prediction done')

#Evaluate the model (Perfect MAE = 0)
#Give out MAE of the prediction set compared to the test set
#MAE in minutes
mse = mean_absolute_error(y_test, y_pred)
print('Mean absolute Error for Extra Trees: ' , mse/60)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Mean squared Error for Extra Trees: ', rmse)

importances = model.feature_importances_
features = X.columns

#Create a DataFrame for visualization
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print('Creating DataFrame done')

#Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Extra Trees')
plt.gca().invert_yaxis()
plt.show()
print('Plotting feature importance done')

#Visualize the results
plt.scatter(y_test,y_pred, marker='.',color='red', label='Predicted vs Actual')  # type: ignore
plt.xlabel('Actual EndTime')
plt.ylabel('Predicted EndTime')
plt.title('Regression Predictions vs Actual Values')
plt.legend()
plt.show()
print('Plotting results done')


X_train_minimized = X_train.drop(["COG", "TH", "shiptype", "EndLongitude", "EndLatitude", "pastTravelTime", "latitudeDistance"], axis=1)
X_test_minimized = X_test.drop(["COG", "TH", "shiptype", "EndLongitude", "EndLatitude", "pastTravelTime", "latitudeDistance"], axis=1)
print('All done')
