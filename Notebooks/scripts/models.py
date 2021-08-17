## Import libraries

# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# [install(p) for p in ['scikit-learn', 'catboost', 'lightgbm', ]]
 
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from .preprocessing import *







## Utils functions for train

skf = RepeatedStratifiedKFold(n_splits = 5,
                              random_state=199)

def metric(x, y):
    """
    Custom metric function
    
    Input:
    x: true target value
    y: predicted value
    
    Output: 
    error: rmse between the real value and the predicted one.    
    """
    
    error = mean_squared_error(x, y, squared=False)
    
    return error

def xgb_predict(estimator,train,label, test,estimator_name, skf, cat, groups):
    """
    Train function for managing overfitting and using the best iteration with xgboost model.
    
    Input:
    estimator: XGBRegressor instance model
    train: Design matrix in pandas dataframe format for the learning phase
    label: Target variable in pandas data series format
    test: Design matrix in pandas dataframe format for the prediction phase
    estimator_name: Name of the model being created
    skf: crossvalidation scheme instance; in this case, RepeatedStratifiedKFold
    cat: Discrete Target Variable for the cross validation step
    groups: Categories also for some cross validation scheme. 
    
    Output:
    val_pred: Numpy array containing predictions for the train set
    test_pred: Numpy array containg predictions for the test set
    estimator_name: model name
    """
    
    mean_train = []
    mean_test_val = []
    test_pred = np.zeros(test.shape[0])
    val_pred = np.zeros(train.shape[0])
    for count, (tr_ind,te_ind) in enumerate(skf.split(train, cat, groups)):                                                     
        x_train,x_test = train.iloc[train_index],train.iloc[test_index]
        y_train,y_test = label.iloc[train_index],label.iloc[test_index]
        print(f'========================Fold{count +1}==========================')
        estimator.fit(x_train, y_train, early_stopping_rounds = 200, eval_metric="rmse",
                           eval_set=[(x_test, y_test)],verbose=2500)
        train_predict = estimator.predict(x_train, ntree_limit = estimator.get_booster().best_ntree_limit)
        test_predict = estimator.predict(x_test, ntree_limit = estimator.get_booster().best_ntree_limit)
        val_pred[test_index] = test_predict
        test_pred+= estimator.predict(test, ntree_limit = estimator.get_booster().best_ntree_limit)
        
        print('\nTesting scores', metric(y_test,test_predict))
        print('\nTraining scores', metric(y_train,train_predict))
        mean_train.append(metric(y_train, train_predict))
        mean_test_val.append(metric(y_test,test_predict))
    print('Average Testing ROC score for 50 folds split:',np.mean(mean_test_val))
    print('Average Training ROC score for 50 folds split:',np.mean(mean_train))
    print('standard Deviation for 50 folds split:',np.std(mean_test_val))
    return val_pred, test_pred, estimator_name


def cat_predict(estimator,train,label,test,estimator_name, cat, groups):
    """
    Train function for managing overfitting and using the best iteration with xgboost model.
    
    Input:
    estimator: CatboostRegressor instance model
    train: Design matrix in pandas dataframe format for the learning phase
    label: Target variable in pandas data series format
    test: Design matrix in pandas dataframe format for the prediction phase
    estimator_name: Name of the model being created
    skf: crossvalidation scheme instance; in this case, RepeatedStratifiedKFold
    cat: Discrete Target Variable for the cross validation step
    groups: Categories also for some cross validation scheme. 
    
    Output:
    val_pred: Numpy array containing predictions for the train set
    test_pred: Numpy array containg predictions for the test set
    estimator_name: model name
    """
    
    mean_train = []
    mean_test_val = []
    test_pred = np.zeros(test.shape[0])
    val_pred = np.zeros(train.shape[0])
    for count, (train_index,test_index) in enumerate(skf.split(train, cat, groups)):
        x_train,x_test = train.iloc[train_index],train.iloc[test_index]
        y_train,y_test = label.iloc[train_index],label.iloc[test_index]
        x_train = np.nan_to_num(x_train)
        y_train = np.nan_to_num(y_train)
        x_test = np.nan_to_num(x_test)
        y_test = np.nan_to_num(y_test)
        
        print(f'========================Fold{count +1}==========================')
        estimator.fit(x_train,y_train,eval_set=[(x_test,y_test)],early_stopping_rounds=200,
                           verbose=2500,use_best_model=True)
        train_predict = estimator.predict(x_train)
        test_predict = estimator.predict(x_test)
        val_pred[test_index] = test_predict
        test_pred+= estimator.predict(test)
        
        print('\nTesting scores', metric(y_test,test_predict))
        print('\nTraining scores', metric(y_train,train_predict))
        mean_train.append(metric(y_train, train_predict))
        mean_test_val.append(metric(y_test,test_predict))
    print('Average Testing ROC score for 50 folds split:',np.mean(mean_test_val))
    print('Average Training ROC score for 50 folds split:',np.mean(mean_train))
    print('standard Deviation for 50 folds split:',np.std(mean_test_val))
    return val_pred, test_pred, estimator_name


def lgb_predict(estimator,train,label,test,estimator_name, cat, groups):
    """
    Train function for managing overfitting and using the best iteration with xgboost model.
    
    Input:
    estimator: LGBMRegressor instance model
    train: Design matrix in pandas dataframe format for the learning phase
    label: Target variable in pandas data series format
    test: Design matrix in pandas dataframe format for the prediction phase
    estimator_name: Name of the model being created
    skf: crossvalidation scheme instance; in this case, RepeatedStratifiedKFold
    cat: Discrete Target Variable for the cross validation step
    groups: Categories also for some cross validation scheme. 
    
    Output:
    val_pred: Numpy array containing predictions for the train set
    test_pred: Numpy array containg predictions for the test set
    estimator_name: model name
    """
    
    mean_train = []
    mean_test_val = []
    test_pred = np.zeros(test.shape[0])
    val_pred = np.zeros(train.shape[0])
    for count, (train_index,test_index) in enumerate(skf.split(train, cat, groups)):
        x_train,x_test = train.iloc[train_index].values,train.iloc[test_index].values
        y_train,y_test = label.iloc[train_index].values,label.iloc[test_index].values
        print(f'========================Fold{count +1}==========================')
        estimator.fit(x_train,y_train,eval_set=[(x_test,y_test)],early_stopping_rounds=200,
                               verbose=2500)
        train_predict = estimator.predict(x_train, num_iteration = estimator.best_iteration_)
        test_predict = estimator.predict(x_test, num_iteration = estimator.best_iteration_)
        val_pred[test_index] = test_predict
        test_pred+= estimator.predict(test, num_iteration = estimator.best_iteration_)
        
        print('\nValidation scores', metric(y_test,test_predict))
        print('\nTraining scores', metric(y_train,train_predict))
        mean_train.append(metric(y_train, train_predict))
        mean_test_val.append(metric(y_test,test_predict))
    print('Average Testing ROC score for 50 folds split:',np.mean(mean_test_val))
    print('Average Training ROC score for 50 folds split:',np.mean(mean_train))
    print('standard Deviation for 50 folds split:',np.std(mean_test_val))
    return val_pred, test_pred, estimator_name

def model_predict(estimator,train,label,test, estimator_name, cat, groups):
    """
    Train function for managing overfitting and using the best iteration with xgboost model.
    
    Input:
    estimator: Regressor instance model
    train: Design matrix in pandas dataframe format for the learning phase
    label: Target variable in pandas data series format
    test: Design matrix in pandas dataframe format for the prediction phase
    estimator_name: Name of the model being created
    skf: crossvalidation scheme instance; in this case, RepeatedStratifiedKFold
    cat: Discrete Target Variable for the cross validation step
    groups: Categories also for some cross validation scheme. 
    
    Output:
    val_pred: Numpy array containing predictions for the train set
    test_pred: Numpy array containg predictions for the test set
    estimator_name: model name
    validation_error: Average error on validation sets
    train_error: Average error on train sets
    """
    
    mean_train = []
    mean_test_val = []
    test_pred = np.zeros((test.shape[0]))
    val_pred = np.zeros((train.shape[0]))
    for count, (train_index,test_index) in enumerate(skf.split(train, cat, groups)):
        x_train,x_test = train.iloc[train_index].values,train.iloc[test_index].values
        y_train,y_test = label.iloc[train_index].values,label.iloc[test_index].values
        print(f'========================Fold{count +1}==========================')
        estimator.fit(x_train, y_train)
        train_predict = estimator.predict(x_train)
        test_predict = estimator.predict(x_test)
        val_pred[test_index] = test_predict.reshape((test_predict.shape[0],))
        test_pred+= estimator.predict(test.values)
        
        print('\nValidation scores', metric(y_test,test_predict))
        print('\nTraining scores', metric(y_train,train_predict))
        mean_train.append(metric(y_train, train_predict))
        mean_test_val.append(metric(y_test,test_predict))
        
        validation_error = np.mean(mean_test_val)
        train_error = np.mean(mean_train)
        
    print('Average Testing RMSE  for 50 folds split:',np.mean(mean_test_val))
    print('Average Training RMSE  for 50 folds split:',np.mean(mean_train))
    print('standard Deviation for 50 folds split:',np.std(mean_test_val))
    return val_pred, test_pred, estimator_name, validation_error, train_error

def Create_StackDataFrames(train_preds, test_preds, names):
    """
    Function to create stacked dataframes of predictions on train and test sets to make Ensembling
    
    Input: 
    train_preds: list of predictions on train set
    test_preds: list of predictions on test set
    names: list of model names for each prediction
    
    Output:
    Train_stack: pandas dataframe of stacked predictions on train set
    Test_stack: pandas dataframe of stacked predictions on test set
    """
    Train_stack = pd.concat([pd.Series(tr_pred, name=name) for tr_pred, name in zip(train_preds, names)],1)
    
    Test_stack = pd.concat([pd.Series(te_pred, name=name) for te_pred, name in zip(test_preds, names)],1)
    
    Test_stack = Test_stack/50 #average predictions for 50 folds on the Test set..
    
    return Train_stack, Test_stack
    
def Stack(meta_estimator,Train_stack,Test_stack,target,file_name, cat, groups, ss):
    """
    Train function for ensembling catboost and lightgbm models
    
    Input:
    meta_estimator: Regressor instance model
    Train_stack: pandas dataframe of stacked train set predictions
    Test_stack: pandas dataframe of stacked test set predictions
    target: Target variable in pandas data series format
    file_name: submission name
    cat: Discrete Target Variable for the cross validation step
    groups: Categories also for some cross validation scheme
    ss: Sample Submission file 
    
    Output:
    ss: SampleSubmission dataframe with predictions added
    val_pred: Numpy array containing predictions for the train set
    test_pred: Numpy array containg predictions for the test set
    estimator_name: model name
    train_score: Average error on validation sets
    test_score: Average error on train sets
    """
    
    val_pred, test_pred, estimator_name, test_score, train_score = model_predict(meta_estimator,Train_stack, target, Test_stack, "Ridge", cat, groups)
    
    prediction = test_pred/50#meta_estimator.fit(Train_stack, target).predict(Test_stack)
    
    ss['Target'] = prediction#np.round(np.absolute(prediction), 0)
    
    ss.Target=ss.Target.apply(abs)
    
    ss.to_csv(file_name,index=False)
    
    return ss, val_pred, test_pred, estimator_name, test_score, train_score

def trainer(model_name, train, test, ss, device_type="CPU"):
    """
    Function to train models and create submisison
    
    Input:
    model_name: name of model
    train: Design matrix in pandas dataframe format for the learning phase
    test: Design matrix in pandas dataframe format for the prediction phase
    device_type: CPU or GPU
    ss: Sample Submission file 
    
    Output:
    ss: SampleSubmission dataframe with predictions added
    val_pred: Numpy array containing predictions for the train set
    test_pred: Numpy array containg predictions for the test set
    estimator_name: model name
    train_score: Average error on validation sets
    test_score: Average error on train sets
    """
    
    ## Preprocessing
    
    xtrain, ytrain, xtest, cat2, cat, groups = preprocessor(train, test) ## Preprocesing data
    
    ## Defining model one
    
    catboost =  CatBoostRegressor(random_seed=34,use_best_model=True,
                          n_estimators=400000,silent=True,eval_metric='RMSE', task_type=device_type,
#                                   learning_rate=0.7
                                 )

    ## Training model one
    cat1_train, cat1_test, cat1_name = cat_predict(catboost,xtrain, ytrain, xtest,  'catboost(1)', cat, groups)
    
    ## Defining second model
    lgb_model = LGBMRegressor(random_state=34, n_estimators=100000, colsample_bytree=0.9, min_child_samples=10, subsample=0.7,subsample_freq=2,num_leaves=120,reg_lambda=1,reg_alpha=1, metric="rmse", 
                              learning_rate=0.01, 
                              max_depth=5, device=device_type.lower())

    ## Training second model
    LGB1__train, LGB1_test, LGB1_name =lgb_predict(lgb_model,xtrain, ytrain, xtest,'lightgbm(1)', cat, groups)
    
    
    ## Stacking predictions of models trained 
    Train_stack1, Test_stack1 = Create_StackDataFrames([cat1_train, LGB1__train], [cat1_test, LGB1_test], [cat1_name, LGB1_name])
    
    ## Defining meta estimator
    meta_estimator = Ridge()
    
    ## Ensembling and making submisison
    sub, val_pred, test_pred, estimator_name, test_score, train_score = Stack(meta_estimator, Train_stack1, Test_stack1, ytrain, f'../Submissions/stack_{model_name}.csv', cat, groups, ss) 

    return sub, val_pred, test_pred, estimator_name, test_score, train_score







