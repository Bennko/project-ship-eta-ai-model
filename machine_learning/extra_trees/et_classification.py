from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier

#create a synthetic binary classification problem with 1000 examples and 20 input features
#define the datatset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,n_redundant=5, random_state=4)

#define the model
model = ExtraTreesClassifier()

#evaluate the model
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

#print classification accouracy of the model on the dataset in percent
#print('Accuracy: %.3f (%.3f)' %(mean(n_scores), std(n_scores)))

#This program creates a sample dataset with 1000 examples and 20 input features
#We evaluate the model using repeated k-fold cross-validation, with 3 repeats and 10 folds
#At the end we give out the mean and standard deviation of the accuracy of the model across all repeats and folds
#The results vary at each run, because of the statistic nature of the algorithm

#In the following, we use the model as a final model and make predictions for classification
#Extra Trees ensemble is fit on all data, then the predict() funtion is called to make a prediction on new data

#fit the model on whole dataset
model.fit(X,y)

#make a single prediction
row = [[-3.52169364,4.00560592,2.94756812,-0.09755101,-0.98835896,1.81021933,-0.32657994,1.08451928,4.98150546,-2.53855736,3.43500614,1.64660497,-4.1557091,-1.55301045,-0.30690987,-1.47665577,6.818756,0.5132918,4.3598337,-4.31785495]]
yhat = model.predict(row)

print('Predicted Class: %d' % yhat[0])