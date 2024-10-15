


#Boosting
#Adaboost

#instructions
'''Import AdaBoostClassifier from sklearn.ensemble.
Instantiate a DecisionTreeClassifier with max_depth set to 2.
Instantiate an AdaBoostClassifier consisting of 180 trees and setting the base_estimator to dt.'''


# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Instantiate dt
dt = DecisionTreeClassifier(max_depth = 2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)


#instructions
'''Fit ada to the training set.
Evaluate the probabilities of obtaining the positive class in the test set.'''

# Fit ada to the training set
ada.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:,1]


#instructions
'''Import roc_auc_score from sklearn.metrics.
Compute ada's test set ROC AUC score, assign it to ada_roc_auc, and print it out.'''

# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))


#Gradient Boosting
#instructions
'''Import GradientBoostingRegressor from sklearn.ensemble.
Instantiate a gradient boosting regressor by setting the parameters:
max_depth to 4
n_estimators to 200'''

# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate gb
gb = GradientBoostingRegressor(max_depth = 4, 
            n_estimators = 200,
            random_state=2)


#instructions
'''Fit gb to the training set.
Predict the test set labels and assign the result to y_pred.'''

# Fit gb to the training set
gb.fit(X_train, y_train)

# Predict test set labels
y_pred = gb.predict(X_test)


#instructions
'''Import mean_squared_error from sklearn.metrics as MSE.
Compute the test set MSE and assign it to mse_test.
Compute the test set RMSE and assign it to rmse_test.'''

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute MSE
mse_test = MSE(y_test,y_pred)

# Compute RMSE
rmse_test = MSE(y_test, y_pred) ** (1/2)

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))


#Stochastic Gradient Boosting (SGB)
#instructions
'''Instantiate a Stochastic Gradient Boosting Regressor (SGBR) and set:
max_depth to 4 and n_estimators to 200,
subsample to 0.9, and
max_features to 0.75.
'''

# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4, 
            subsample= 0.9,
            max_features=0.75,
            n_estimators=200,
            random_state=2)



#instructions
'''Fit sgbr to the training set.
Predict the test set labels and assign the results to y_pred.'''

# Fit sgbr to the training set
sgbr.fit(X_train, y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)


#instructions
'''Import mean_squared_error as MSE from sklearn.metrics.
Compute test set MSE and assign the result to mse_test.
Compute test set RMSE and assign the result to rmse_test.'''

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error

# Compute test set MSE
mse_test = MSE(y_test, y_pred)

# Compute test set RMSE
rmse_test = mse_test ** (1/2)

# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))



#Model tuning
#tuning a carts hyperparameter

#instructions
'''Define a grid of hyperparameters corresponding to a Python dictionary called params_dt with:
the key 'max_depth' set to a list of values 2, 3, and 4
the key 'min_samples_leaf' set to a list of values 0.12, 0.14, 0.16, 0.18'''

# Define params_dt
params_dt = {'max_depth': [2,3,4],
'min_samples_leaf': [0.12,0.14,0.16,0.18]}


#instructions
'''Import GridSearchCV from sklearn.model_selection.
Instantiate a GridSearchCV object using 5-fold CV by setting the parameters:
estimator to dt, param_grid to params_dt and
scoring to 'roc_auc'.'''

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='roc_auc',
                       cv=5,
                       n_jobs=-1)



#instructions
'''Import roc_auc_score from sklearn.metrics.
Extract the .best_estimator_ attribute from grid_dt and assign it to best_model.
Predict the test set probabilities of obtaining the positive class y_pred_proba.
Compute the test set ROC AUC score test_roc_auc of best_model.'''

# Import roc_auc_score from sklearn.metrics
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))



#Tuning an RF's hyperparameter
#instructions
'''Define a grid of hyperparameters corresponding to a Python dictionary called params_rf with:
the key 'n_estimators' set to a list of values 100, 350, 500
the key 'max_features' set to a list of values 'log2', 'auto', 'sqrt'
the key 'min_samples_leaf' set to a list of values 2, 10, 30'''

# Define the dictionary 'params_rf'
params_rf = {'n_estimators': [100,350,500],
'max_features': ['log2', 'auto','sqrt'],
'min_samples_leaf':[2,10,30]}


#instructions
'''Import GridSearchCV from sklearn.model_selection.
Instantiate a GridSearchCV object using 3-fold CV by using negative mean squared error as the scoring metric.'''

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator= rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)



#instructions
'''Import mean_squared_error as MSE from sklearn.metrics.
Extract the best estimator from grid_rf and assign it to best_model.
Predict best_model's test set labels and assign the result to y_pred.
Compute best_model's test set RMSE.'''

# Import mean_squared_error from sklearn.metrics as MSE 
from sklearn.metrics import mean_squared_error as MSE 

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test)) 