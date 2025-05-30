import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# Function for the time difference
def trip_time(start_lat, start_lon, end_lat, end_lon):
# Trying to implement the calculation of the difference between the start and endpoint of a specific trip
    pass
# Load csv file
data = pd.read_csv("felixstowe_rotterdam.csv")

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the KNeighborsRegressor model
knn_regressor = KNeighborsRegressor()
knn_regressor.fit(X_train, Y_train)

# Make predictions on the testing set
Y_pred = knn_regressor.predict(X_test)

# Calculate mean absolute error to evaluate the model
mae = mean_absolute_error(Y_test, Y_pred)
print("Mean Absolute Error:", mae)