import pandas as pd
import numpy as np

from datamining._errors import _validate_method_argument_types, _validate_class_argument_types
from datamining.check import check_numeric_data, check_category_data, check_time_series_data, check_time_interval_data, check_data
from datamining.transformations import standarization, normalization_min_max, log_transformation, \
    transformation_box_cox, root_transformation, one_hot_encoding, binarization
from datamining.regression import BestSimpleLinearRegression, BestMultipleLinearRegression
from datamining.preprocessing import handle_missing_numeric, handle_missing_categories
from datamining.regression import BestSimpleLinearRegression, BestMultipleLinearRegression
from datamining.classification import BestClassification

np.random.seed(42)
random_start_dates = pd.to_datetime(np.random.choice(pd.date_range('2022-01-01', '2023-12-31'), size=10, replace=False))
random_end_dates = random_start_dates + pd.to_timedelta(np.random.randint(1, 30, size=10), unit='D')
random_intervals_l = pd.to_timedelta(np.random.randint(1, 60, size=10), unit='D')
random_intervals_m = pd.to_timedelta(np.random.randint(1, 120, size=10), unit='D')

ramka_testowa = pd.DataFrame({'a': [3, 4, 5, 6, 3, 3, -3, 3, 3, 32],
                              'b': [2.1, 23, 2.1, 2.1, 2.1, -88.2, 2.23, -8.99, 2.2, 2.23],
                              'c': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              'd': [1.3, 2.9, 3.1, 3.5, 5, 6, 7, 8, 11, 9.9],
                              'e': [2, 3, 4, 3, 5, 6, 7, 8, 9, 10],
                              'f': [True, False, False, False, True, True, False, True, True, True],
                              'g': pd.Categorical(['x', 'z', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 's']),
                              'h': pd.Categorical([1, 2, 2, 2, 2, 2, 1, 1, 3, 3]),
                              'i': pd.Categorical(['a', 'c', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']),
                              'j': random_start_dates,
                              'k': random_end_dates,
                              'l': random_intervals_l,
                              'm': random_intervals_m
                              })


# Opis podmodulu _errors
'''
@_validate_method_argument_types
def test(a: int):
    return a


#print(test(1))
print(test('a'))
'''

'''
class Test:
    def __init__(self, a: int):
        self._a = a
        _validate_class_argument_types({"a": ["int", self._a]})
        print(a)


instance = Test(1)
instance = Test('a')
'''

# Opis podmodulu check
'''
print(check_numeric_data(df=ramka_testowa, round=2, use=False))
print(check_category_data(df=ramka_testowa, use=False, cat_dist=True))
print(check_time_series_data(df=ramka_testowa, use=False))
print(check_time_interval_data(df=ramka_testowa, use=False))
check_data(df=ramka_testowa)
'''

# Opis podmodulu normalizations
'''
df = pd.DataFrame({
    'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'b': [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
})

standarization(df)
normalization_min_max(df, ['a', 'b'])
print(log_transformation(df, ['a', 'b'], 2.0,10.0,1.0))
print(transformation_box_cox(df, columns=['a', 'b'], alpha=3))
print(root_transformation(df, ['a', 'b'], root=2))
print(binarization(df, ['a', 'b'], border=8, values=['<8', '>=8']))
df = pd.DataFrame({
    'Category': pd.Categorical(['x', 'z', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 's'])
})
print(one_hot_encoding(df, columns=['Category'], values=[0, 1], prefix_sep='_', drop=True))
'''

# Opis podmodulu preprocessing
'''
df = pd.DataFrame({
    'a': [None, 2, None, 4, None, 6, 7, 8, 9, 10],
    'b': [None, None, 9, 16, 25, None, 49, 64, 81, None]
})
print(handle_missing_numeric(df, columns=['a', 'b'], strategy='median'))

df = pd.DataFrame({
    'c': pd.Categorical(['x', None, 'z', 'x', 'y', 'z', 'x', 'y', 'z', 's']),
    'd': pd.Categorical(['x', 'z', None, 'x', 'y', 'z', 'x', 'y', 'z', 's'])
})
print(handle_missing_categories(df, columns=['c', 'd'], strategy='drop'))
'''

# Opis podmodulu regression
"""
instance = BestMultipleLinearRegression(ramka_testowa,
                                        response='c',
                                        set_seed=13,
                                        divide_method='crossvalidation',
                                        k=5)

print(f"best_feature: {instance.best_features}")
print(f"model: {instance.model}")
print(f"formula: {instance.formula}")
print(f"intercept: {instance.intercept}")
print(f"coefficient: {instance.coefficients}")
print(f"r_squared: {instance.r_squared}")
print(f"y_pred: {instance.y_pred}")
print(f"rmspe: {instance.rmspe}")
print(f"rss: {instance.rss}")
print(f"mape: {instance.mape}")
print(f"mse: {instance.mse}")
print(f"rmse: {instance.rmse}")
instance.plot_model()
"""

# Opis podmodulu classification

ramka_testowa = pd.DataFrame({
    'a': [3, 4, 5, 6, 3, 3, -3, 3, 3, 32, 23, 4, 5, 6, -2, 34, 12, 54, 1, 17],
    'b': [True, False, False, False, True, True, False, True, True, True,
          True, False, False, False, True, True, False, True, True, True],
    'c': pd.Categorical(['x', 'z', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 's',
                         'x', 'z', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 's']),
    'd': pd.Categorical([1, 2, 2, 2, 2, 2, 1, 1, 3, 3, 1, 1, 3, 3, 2, 3, 2, 2, 1, 3]),
    'response': pd.Categorical(['a', 'c', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a',
                                'a', 'c', 'c', 'c', 'b', 'a', 'a', 'b', 'b', 'a'])
})

instance = BestClassification(df=ramka_testowa,
                              response='response',
                              set_seed=123,
                              divide_method='train_test',
                              test_size=0.25)

for attribute, value in instance.__dict__.items():
    print(f"{attribute}: {value}")

print(instance.accuracy)
