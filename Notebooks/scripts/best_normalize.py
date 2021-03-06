#https://github.com/miguelcdpmarques/BestNormalize/blob/master/best_normalize.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer, quantile_transform
from scipy.stats import normaltest, boxcox
import math
import IPython


class BestNormalize(BaseEstimator, TransformerMixin):
    
    def __init__(self, standardize=True, variables='all', allow_quantile=True, plot=False, verbose=False): 
        # no *args or **kwargs
        self.standardize = standardize
        self.variables = variables
        self.allow_quantile = allow_quantile
        self.scores = []
        self.transformation_margins = {}
        # if plot: plot both the histogram with the normal fit line and the QQ plots.
        # if verbose: print information about the test statistics
        
    def fit(self, X, y=None):
        self.init_scores(X)
        X = X.select_dtypes(include=np.number)
        for column in X.columns:
            data = X[column].dropna().copy()
            self.scores.extend([[column] + self.test_transformation_log(data)])
            #self.scores.extend([[column] + self.test_transformation_sqrt(data)])
            #self.scores.extend([[column] + self.test_transformation_exp(data)])
            #self.scores.extend([[column] + self.test_transformation_arcsin(data)])
            #self.scores.extend([[column] + self.test_transformation_yeo_johnson(data)])
            #self.scores.extend([[column] + self.test_transformation_boxcox(data)])
            if self.allow_quantile:
                self.scores.extend([[column] + self.test_transformation_quantile(data)])
        self.scores = pd.DataFrame(data=self.scores, 
                        columns=['Variable','Transformation','Score','Approx. Normal?'])
        return self
    
    def transform(self, X, y=None):
        map_transformation = {'original': self.apply_no_transformation, 'log': self.apply_transformation_log,
                    'sqrt': self.apply_transformation_sqrt, 'exp': self.apply_transformation_exp,
                    'arcsin': self.apply_transformation_arcsin, 
                    'yeo-johnson': self.apply_transformation_yeo_johnson,
                    'boxcox': self.apply_transformation_boxcox, 'quantile':self.apply_transformation_quantile}
        for column in X.columns:
            transformation = self.best_scores[self.best_scores['Variable']==column]['Transformation'].iloc[0]
            X[column] = map_transformation[transformation](X[column])
        if self.standardize:
            self.scaler = StandardScaler()
            X = pd.DataFrame(data=self.scaler.fit_transform(X), columns=X.columns)
        return X
    
    def inverse_transform(self, X):
        if self.standardize:
            X = pd.DataFrame(data=self.scaler.inverse_transform(X), columns=X.columns)
        map_inverse_transformation = {'original': self.apply_no_transformation, 
                    'log': self.apply_inverse_transformation_log}
        for column in X.columns:
            transformation = self.best_scores[self.best_scores['Variable']==column]['Transformation'].iloc[0]
            X[column] = map_inverse_transformation[transformation](X[column])
        return X

    def init_scores(self, X):
        for column in X.columns:
            data = X[column].dropna()
            self.scores.extend([[column, 'original'] + self.gaussian_metric(data)])
            
    def gaussian_metric(self, data):
        alpha = 0.05
        stat, p = normaltest(data)
        if isinstance(stat, np.ndarray):
            stat, p = stat.item(), p.item()
        test_score = [stat, (p > alpha)]
        return test_score
    
    @property
    def best_scores(self):
        return self.scores.iloc[self.scores.groupby('Variable')['Score'].idxmin()]
    
    def test_transformation_log(self, data):
        a = 1
        if min(data) <= 0:
            a = (min(data) * -1) + 1
        _data = np.log(data + a)
        return ['log'] + self.gaussian_metric(_data)
    
    def test_transformation_sqrt(self, data):
        a = 1
        if min(data) <= 0:
            a = (min(data) * -1) + 1
        _data = np.sqrt(data + a)
        return ['sqrt'] + self.gaussian_metric(_data)
    
    def test_transformation_exp(self, data):
        try:
            _data = data.apply(lambda x: math.exp(x))
        except OverflowError:
            return ['exp', None, False]
        return ['exp'] + self.gaussian_metric(_data)

    def test_transformation_arcsin(self, data):
        _data = np.log([i + np.sqrt(i**2 + 1) for i in data])
        return ['arcsin'] + self.gaussian_metric(_data)

    def test_transformation_yeo_johnson(self, data):
        pt = PowerTransformer(method='yeo-johnson', standardize=False, copy=True)
        _data = pt.fit_transform(np.array(data).reshape(-1,1))
        return ['yeo-johnson'] + self.gaussian_metric(_data)

    def test_transformation_boxcox(self, data):
        if min(data) < 0:
            return ['boxcox', None, False]
        else:
            _data, _ = boxcox(data)
            return ['boxcox'] + self.gaussian_metric(_data)

    def test_transformation_quantile(self, data):
        _data = quantile_transform(np.array(data).reshape(-1,1), output_distribution='normal')
        return ['quantile'] + self.gaussian_metric(_data)

    @staticmethod
    def apply_no_transformation(x):
       return x

    def apply_transformation_log(self, x):
        self.transformation_margins[x.name] = 1
        if min(x) <= 0:
            self.transformation_margins[x.name] = (min(x) * -1) + 1
        return np.log(x + self.transformation_margins[x.name])
    
    def apply_inverse_transformation_log(self, x):
        return np.exp(x - self.transformation_margins[x.name])
        
    @staticmethod
    def apply_transformation_sqrt(x):
        a = 1
        if min(x) <= 0:
            a = (min(x) * -1) + 1
        return np.sqrt(x + a)

    @staticmethod
    def apply_transformation_exp(x):
        return x.apply(lambda i: math.exp(i))

    @staticmethod
    def apply_transformation_arcsin(x):
        return np.log(x + np.sqrt(x**2 + 1))

    @staticmethod
    def apply_transformation_yeo_johnson(x):
        pt = PowerTransformer(method='yeo-johnson', standardize=False, copy=True)
        return pt.fit_transform(np.array(x).reshape(-1,1))

    @staticmethod
    def apply_transformation_boxcox(x):
        if min(x) < 0:
            raise ValueError('You might be normalizing twice. The values of the input data must be strictly positive to apply the boxcox transformation.')
        return boxcox(x)[0]

    @staticmethod
    def apply_transformation_quantile(x):
        return quantile_transform(np.array(x).reshape(-1,1), output_distribution='normal')


### mention transformation applied in each case

if __name__ == '__main__':
    housing = pd.read_csv('datasets/housing.csv')
    housing_num = housing.select_dtypes(include=np.number).drop(['latitude','longitude'], axis=1)

    bestNormalize = BestNormalize(allow_quantile=False)
    housing_num_transformed = bestNormalize.fit_transform(housing_num)
    housing_num_inverse = bestNormalize.inverse_transform(housing_num_transformed)
    print(bestNormalize.best_scores)
    print(np.allclose(housing_num, housing_num_inverse))
    print(housing_num[:5])
    print(housing_num_inverse[:5])