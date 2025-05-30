import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.ensemble import ExtraTreesRegressor


data = pd.read_csv("rotterdam_hamburg_clean_new.csv", on_bad_lines="warn")
#data.insert(0, "pastTravelTime",(pd.to_datetime(data["time"]) - pd.to_datetime(data["StartTime"])).dt.total_seconds())
data.insert(0, "longitudeDistance", data["EndLongitude"] - data["StartLongitude"])
data.insert(0, "latitudeDistance", data["EndLatitude"] - data["StartLatitude"])

test_features = [ "COG", "TH", "shiptype", "MMSI", "TripID", "EndLongitude", "EndLatitude", "pastTravelTime", "longitudeDistance", "latitudeDistance"]

y = (pd.to_datetime(data["EndTime"]) - pd.to_datetime(data["time"])).dt.total_seconds()
X = data[["Latitude", "Longitude", "SOG"] + test_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


adaboost_regressor = AdaBoostRegressor(n_estimators=100, random_state=42)
adaboost_regressor.fit(X_train, y_train)
y_predict = adaboost_regressor.predict(X_test)
mse_decision_trees = mean_absolute_error(y_test, y_predict)
print("Mean absolute Error for Decision Trees: " , mse/60)

base_regressor_knn = KNeighborsRegressor(n_neighbors=1)
adaboost_regressor = AdaBoostRegressor(estimator=base_regressor_knn, n_estimators=50, random_state=42)
adaboost_regressor.fit(X_train, y_train)
y_predict_decision = adaboost_regressor.predict(X_test)
mse_knn = mean_absolute_error(y_test, y_predict)
print("Mean absolute Error for KNN: " , mse/60)


base_regressor_svr = make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5, max_iter=10000, dual='auto'))
adaboost_regressor = AdaBoostRegressor(estimator=base_regressor_svr, n_estimators=50, random_state=42)
adaboost_regressor.fit(X_train, y_train)
y_predict = adaboost_regressor.predict(X_test)
mse_linear_svr = mean_absolute_error(y_test, y_predict)
print("Mean absolute Error for Linear SVR: " , mse/60)


base_regressor_lr = linear_model.LinearRegression()
adaboost_regressor = AdaBoostRegressor(estimator=base_regressor_lr, n_estimators=50, random_state=42)
adaboost_regressor.fit(X_train, y_train)
y_predict_linear = adaboost_regressor.predict(X_test)
mse_linear = mean_absolute_error(y_test, y_predict)
print("Mean absolute Error for Linear Regression: " , mse/60)
