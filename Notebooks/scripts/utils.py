import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#Handling date features
def date_features(df):
    date_cols = [col for col in df.columns if 'Date' in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    for col in date_cols:
        for attr in ['day', 'month', 'week', 'dayofweek', 'weekofyear', 'days_in_month', 'is_month_start', 'is_month_end', 'dayofyear']:
            df[f"{col}_{attr}"] = getattr(df[col].dt, attr)
        df[f"{col}_is_weekend"] = ((df[f'{col}_dayofweek'] >= 5)*1).astype(int)
        df[f"{col}_fortnight"] = (df[f"{col}_day"]%15).astype(int)
        df[f"{col}_which_fortnight"] = (df[f"{col}_day"]//15).astype(int)
        
    df.drop(date_cols, axis=1, inplace=True)
    return df

## Data reader
def data_reader(path):
    files = [x for x in os.listdir(path) if x.endswith("csv")]
    tr_name = [x for x in files if "train" in x.lower()][0]
    te_name = [x for x in files if "test" in x.lower()][0]
    ss_name = [x for x in files if "submission" in x.lower()][0]
    
    train = pd.read_csv(f"{path}/{tr_name}")
    test = pd.read_csv(f"{path}/{te_name}")
    ss = pd.read_csv(f"{path}/{ss_name}")
    return train, test, ss

## Missing values explorer
def explore_missings(df):
    # A plot to check percentage of missing values for each column in the train data

    ax_train = df.isna().sum().transform(lambda x:x/df.shape[0]).sort_values().plot.barh(figsize=(16, 12), title='% missing values ', color='purple', alpha=.8)
    for i in ax_train.patches:
        ax_train.text(i.get_width()+0.005, i.get_y(), str(int(round(i.get_width(), 2)*100)) + '%', fontsize=13, color='red', alpha=1)
        

def filling_missing_values_and_split(train, test, ID_cols, target_name):
    obj_cols = train[train.isna().sum()[train.isna().sum()!=0].index.tolist()].select_dtypes('object').columns.tolist()


    float_cols = train[train.isna().sum()[train.isna().sum()!=0].index.tolist()].select_dtypes('float').columns.tolist()
    
    train["mean Target"] =  train.groupby(['country', 'urban_or_rural']).Target.transform(np.mean)
    train["min Target"] =  train.groupby(['country', 'urban_or_rural']).Target.transform(np.min)
    train["max Target"] =  train.groupby(['country', 'urban_or_rural']).Target.transform(np.max)
    train["variance Target"] =  train.groupby(['country', 'urban_or_rural']).Target.transform(np.var)
    for q in [x for x in np.arange(0,1, 0.25) if x!=0]:
        train[f"quantile {q} Target"] =  train.groupby(['country', 'urban_or_rural']).Target.transform(lambda x: np.quantile(x, q=q))
        
    agg_target_cols=[col for col in train.columns if (col.endswith("Target") and col !="Target")]
    
    test=test.join(train.groupby(["urban_or_rural", "year"])[agg_target_cols].mean(), on=["urban_or_rural", "year"])
    
    
    for col in obj_cols:
        train[col].fillna(train[col].mode().values[0], inplace=True)
        test[col].fillna(test[col].mode().values[0], inplace=True)
        
    for col in float_cols:
        train[col].fillna(train[col].mean(), inplace=True)
        test[col].fillna(test[col].mean(), inplace=True)


    cat_cols = [col for col in train.select_dtypes("object").columns if col != "ID"]
   

    # Encode categorical features
    # Combine train and test set
    ntrain = train.shape[0] # to be used to split train and test set from the combined dataframe
    all_data = pd.concat((train, test)).reset_index(drop=True)
    
    
    float_cols = [col for col in train.select_dtypes(["float"]).columns.tolist() if col!="Target"]
    
    #float_cols = all_data.select_dtypes(["float"]).columns.tolist()
    
    all_data[[f"mean {col}" for col in float_cols]] =  all_data.groupby(['country', 'urban_or_rural'])[float_cols].transform(np.mean)
    all_data[[f"min {col}" for col in float_cols]] =  all_data.groupby(['country', 'urban_or_rural'])[float_cols].transform(np.min)
    all_data[[f"max {col}" for col in float_cols]] =  all_data.groupby(['country', 'urban_or_rural'])[float_cols].transform(np.max)
    all_data[[f"variance {col}" for col in float_cols]] =  all_data.groupby(['country', 'urban_or_rural'])[float_cols].transform(np.var)
    for q in [x for x in np.arange(0,1, 0.25) if x!=0]:
        all_data[[f"quantile {q} {col}" for col in float_cols]] =  all_data.groupby(['country', 'urban_or_rural'])[float_cols].transform(lambda x: np.quantile(x, q=q))
    all_data = pd.get_dummies(data = all_data, columns = cat_cols)
    
    all_data = date_features(all_data)
    
    # Separate train and test data from the combined dataframe
    train = all_data[:ntrain]
    test = all_data[ntrain:]

    main_cols = train.columns.difference([f"{target_name}"]).difference(ID_cols)
    train_ = train[main_cols]
    target = train[f"{target_name}"]
    train = train_
    test = test[main_cols]
    
    #Scaling data
    #scaler = StandardScaler()
    #scaler.fit(train[float_cols])
    #train[float_cols] = scaler.transform(train[float_cols])
    #test[float_cols] = scaler.transform(test[float_cols])
    
    
    
    return train, test, target