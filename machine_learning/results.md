##  Machine learning feature selection
The following features were selected and used for all the machine learning models we considered:

1. Latitude
2. Longitude
3. SOG
4. COG
5. EndLongitude
6. TH
7. shiptype
8. EndLatitude
9. pastTravelTime

It is important to note that the pastTravelTime feature was added by us and it represents the time difference between the 
current time at which the AIS Data was sent and the start time. 
Initially we tried out different combinations of the features 1-9 and even included some other columns not listed here eg. TripID and MMSI just to make notice of how various parameters affect the prediciton of the model.
We then settled on the features 1-9 as input features and experimented on the different
algorithms by changing their parameters to observe how the mean average errors kept changing. Upon identifying a trend in the shift of the mean avg. errors with respect to parameters such as "n_estimators", "max_depth", we then settled with the parameters which have been provided clearly in the respective sections of each algorithm. 




## Machine learning testing results

After researching for possible candidates to solve the ETT problem we decided to test the following algorithms:
- random forests
- extra trees
- Adaptive Boosting (with KNN)
- Adaptive Boosting (with Linear regression)
- Neural network with gradient descent
- Linear regression (reference)
- XG Boost 

We evaluated each algorithm by looking at the mean absolute error (mae), since it gives a good idea of the quality of the model and is easily enterpretable, since it represents how many seconds/minutes the model predicted over or under the actual result.

For testing we primarily used the trip from Rotterdam to Hamburg as reference although in our testing process we also tested the trip from Felixstowe to Rotterdam.
- Trip length: 18h - 24h

### Random forests
- Final test result for mae: 8.64 minutes
- Features:  "Latitude", "Longitude", "SOG", "COG", "TH", "shiptype", "EndLongitude", "EndLatitude", "pastTravelTime"

Description:

Using the feature importance function we visualized and isolated features to be more irrelevant than others. Based on that we than tested a minimized model with less features that performed comparabel but slightly worse at a mae of 8.64 minutes.

=> Considering the length of the trip it is safe to say that the accuracy of this algorithm is satisfying

### Extra trees
- Final test result for mae: 5.73 minutes
- Features: "Latitude", "Longitude", "SOG", "COG", "TH", "shiptype", "EndLongitude", "EndLatitude", "pastTravelTime"

Description:  
n_estimators: number of trees in forest  
-> Default: 100  
max_features: The number of features to consider when looking for the best split  
-> Default: auto  
min_samples_split: The minimum number of samples required to split an internal node  
-> Default: 2  
min_samples_leaf: The minimum number of samples required to be at a leaf node  
-> Default: 1  
max_depth: The maximum depth of the tree  
-> Default: None (nodes are expanded until all leaves contain less than min_samples_split samples)  

n_estimators must be high but cant be too high, otherwise the runtime is too high
best so far is 350

I kept max_features at 0.5 

min_samples_leaf and max_depth on default

Best results with auto min_samples_split -> 10

Best result: MAE: 1.8  
n_estimators=350  
max_features=auto  
min_samples_split=2  
min_samples_leaf=1  
max_depth=None  
random_state=42  
 

### Adaptive boosting (KNN)
- Final test result for mae: 29.95 minutes
- Features: "Latitude", "Longitude", "SOG", "COG", "TH", "shiptype", "MMSI", "TripID", "EndLongitude", "EndLatitude", "pastTravelTime"

Description:

=> We used the KNN regressor as a base model for the Adaptive boosting algorithm here. We tested this algorithm with k=1, k=5, and k=10 neighbors. We observed high mean absolute errors for higher values of k and this could be a result of high bias in the algorithm. Overall the best mae was recorded with k=1 which was also the fastet time observed when training the model.

### Adaptive boosting (Linear regression)
- Final test result for mae: 54.95 minutes
- Features: "Latitude", "Longitude", "SOG", "COG", "TH", "shiptype", "MMSI", "TripID", "EndLongitude", "EndLatitude", "pastTravelTime"

Description:

=> For this test model, we set the base model to Linear regression by using the Linear regression model from sklearn python library. The algorithm takes no extra parameters and was then trained as such. We observed a high mae indicating that this model is not suitable for our datasets. However, when compared to the others, this model had the fastest training time.

### Neural network with gradient descent
- Final test result for mae: 32.43 minutes
- Features: "Latitude", "Longitude", "SOG", "COG", "TH", "shiptype", "MMSI", "TripID", "EndLongitude", "EndLatitude", "pastTravelTime"
- Epochs: 50

Description:

For the test model we had an input layer with 64 nodes followed by a 32 node hidden layer and a one node output layer. For the optimization we used gradient descent where we picked adam as a reliable and often used option for neural networks. 

=> Considering the length of the trip this result is not very good but might be still acceptable.



## The Broker Agent (Stacking)
We implemented the broker agent which is responsible for serving the requests of clients that want to have an ETT prediction of a vessel by using a method known as stacking. 
The stacking method acts as a broker agent by incorporating the previous predictions of the selected based models to be used as an input training
set for train a so-called meta model. This meta model is repsonsible for the final predictions.

### Selection of Base models (predictor agents)

In the stacking algorithm the following base models were chosen from the models we tested:
1. Random Forest Algorithm
2. Extra Trees Algorithm
3. Adaptive Boosting Algorithm with Decision Trees (Default parameter).

All parameters used for the stacking algorithm remain the same as stated earlier.

### Selection of Meta model
Initially we used the Extra Trees regressor with parameters below as our meta model for final predictions based on the fact that Extra trees had the best results during our testing.

n_estimators=350

min_samples_split=2

 min_samples_leaf=1 

 max_depth=None

 random_state=42

### Repeated K-Fold cross validator

Here we used the K-Fold cross validator to train each base model and validate them  on different subsets of the training data in order to improve the predictions of the base models. We set k=5 thereby splitting the dataset into 5 equal subsets for the algorithm.
All base models were trained on each subset of the split data for each fold.
All base predictions are made on the validation subset of each fold.

Parameters used:

**train_idx**: Indices for the training set in the current fold.

**valid_idx**: Indices for the validation set in the current fold.

Next we average the test set predicitons for some of the following reasons to provide a better prediction:

**Better Cross-Validation**: Since each of the K-Folds train a slightly different version of the model because the training split training data will vary in each fold a better cross-validation is needed. By using predictions from all the base models we are able to provide a better generalized form of prediction over the entire dataset.

**Reduce Overfitting to prevent bias**: Averaging the predictions from multiple models reduces the risk of overfitting to the training subset. This ensures that the final predictions are not overly biased by any particular fold. 

### Creating Meta-Features:

**X_train_meta**:  creates a new training dataset by stacking the predictions from the base models. Each column represents the predictions from one of the base models.
    
**X_test_meta**: Similarly, this creates a new test dataset by stacking the test predictions from the base models.

The resulting matrices X_train_meta and X_test_meta have the same number of rows as X_train and X_test respectively, but the number of columns corresponds to the number of base models.


### Training the Meta Model

We then fit the meta model with X_train_meta and y_train datasets before making our final predictions using the X_test_meta dataset.