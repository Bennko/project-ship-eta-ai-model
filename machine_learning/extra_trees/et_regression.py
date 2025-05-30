#Here we use Extra Trees for a regression problem
#We use the make_regression() function to create a synthetic regression problem with 1000 examples and 20 input features
#We evaluate the model using repeated k-fold cross-validation with three repeats and 10 folds
#At the end we report the mean absolute error of the model across all repeats and folds
#In the scikit library, the MAE is negative so it is maximized instead of minimized. A perfect model has a MAE of 0

from numpy import mean
from numpy import std
from sklearn.datasets import make_regression 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import ExtraTreesRegressor

#define the dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=3)

#define the model
model = ExtraTreesRegressor()

#evaluate the model
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

#report the performance: Give out models mean and standard deviation
#print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


#------------#
#in another approach we fit the model on the data and then use the predict() function to make predictions on new data
#fit the model
model.fit(X,y)

#make a single prediction
row = [[-0.56996683,0.80144889,4.77523539,1.32554027,-1.44494378,-0.80834175,-0.84142896,0.57710245,0.96235932,-0.66303907,-1.13994112,0.49887995,1.40752035,-0.2995842,-0.05708706,-2.08701456,1.17768469,0.13474234,0.09518152,-0.07603207]]
yhat = model.predict(row)
print('Prediction: %d' % yhat[0])

#Hyperparameters for tuning the model
#1: Number of decision trees used in ensemble 
#Set with the n_estimators argument (default: 100)

#2: Number of Features (that is randomly sampled for each split point)