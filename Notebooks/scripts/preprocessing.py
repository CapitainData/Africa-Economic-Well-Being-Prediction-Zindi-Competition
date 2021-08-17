## Import libraries

# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# [install(p) for p in ['scikit-learn', 'kaggler', 'category_encoders', ]]
 
from sklearn.preprocessing import (RobustScaler, PowerTransformer, 
                                   QuantileTransformer, FunctionTransformer, 
                                   MinMaxScaler, StandardScaler, MaxAbsScaler)
from kaggler.preprocessing import FrequencyEncoder
from category_encoders import CountEncoder
import numpy as np
import pandas as pd


## Define functions
def discretizer(label):
    """
    Function to recode Target continuous values in discrete values
    Input:
    label: target continuous value
    
    Output: target discrete value
    """
    if label <=0.2:
        return 1
    elif label <=0.4:
        return 2
    elif label <=0.6:
        return 3
    elif label <=0.8:
        return 4
    else:
        return 5

def preprocessor(train, test, normalizer=RobustScaler(), encoder=CountEncoder(normalize=True)):
    """
    Function to preprocess data befolre modeling
    
    Input:
    train: pandas dataframe of train set
    test: pandas dataframe of test set
    normalizer: Instance of sklearn or similar scaling and centering instance like StandardScaler,
                default, RobustScaler
    encoder: Instance of sklearn or similar Encoder instance for catgorical variables like CountEncoder.
    
    Output:
    xtrain: pandas dataframe of train set preprocessed 
    ytrain: pandas data series of Target variable
    xtest: pandas dataframe of test set preprocessed
    cat: pandas data series of categories of country x urban_or_rural x year for train set
    cat2: pandas data series of categories of Target variable discretized for train set
    groups: pandas data series of countries for train set
    """
    
    to_scale =["ghsl_pop_density", "nighttime_lights", "dist_to_shoreline",
               "dist_to_capital"]
    
    to_scale_100 = ["landcover_crops_fraction", "landcover_urban_fraction", 
                    "landcover_water_permanent_10km_fraction", "landcover_water_seasonal_10km_fraction"]
    
    train[to_scale_100]=train[to_scale_100]/100
    test[to_scale_100]=test[to_scale_100]/100
    
    normalizer.fit(train[to_scale])
    train[to_scale]=normalizer.transform(train[to_scale])
    test[to_scale]=normalizer.transform(test[to_scale])
    
    cat2=train["Target"].apply(discretizer)
    
    ntrain=train.shape[0]
    train_ids=train.ID.unique()
    test_ids=test.ID.unique() 
    
    train.year = train.year.astype(str)
    
    cat = train[["country", "urban_or_rural", "year"]].apply(lambda x: " x ".join(x.values), axis=1)
    
    groups = train["country"]
    
    all_data=pd.concat([train, test], axis=0)
    
    cat_cols = ["country", "urban_or_rural", "year"]
    float_cols = train.columns.difference(cat_cols+["Target", "ID"])
    
    all_data.year=all_data.year.apply(str)
    
    all_data[cat_cols]=encoder.fit_transform(all_data[cat_cols])
    
    
    train = all_data.loc[all_data.ID.isin(train_ids)]
    test = all_data.loc[all_data.ID.isin(test_ids)]
    
    main_cols=train.columns.difference(["ID", 'Target'])
    xtrain = train[main_cols]
    xtest = test[main_cols]
    ytrain=train.Target 
    
    return xtrain, ytrain, xtest, cat, cat2, groups