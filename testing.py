import numpy as np
import pandas as pd
from datamining.preprocessing import handle_numeric_NaN, handle_category_NaN
from datamining.normalizations import (standarization, normalization_min_max, log_transformation,
                                       normalization_box_kox, root_transformation, binarization, one_hot_encoding)
from datamining.check import (check_numeric_data, check_category_data, check_time_series_data, check_time_interval_data,
                              check_data)
from datamining.regression import BestSimpleLinearRegression, BestMultipleLinearRegression
from datamining.classification import BestClassification

pd.set_option('future.no_silent_downcasting', True)

df = pd.DataFrame({'a': [3, 4, 5, 6, 3, 3, -3, 3, 3, 32],
                   'b': [2.1, 23, 2.1, 2.1, 2.1, -88.2, 2.23, -8.99, 2.2, 2.23],
                   'l': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'c': [1.3, 2.9, 3.1, 3.5, 5, 6, 7, 8, 11, 9.9],
                   'd': [2, 3, 4, 3, 5, 6, 7, 8, 9, 10],
                   'e': ["a", "b", "a", "b", "None", "b", "a", "b", "a", "a"],
                   'f': pd.Categorical(['x', 'z', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 's']),
                   'g': pd.Categorical([1, 2, 2, 2, 2, 2, 1, 1, 3, 3]),
                   'h': pd.Series(['foo', 'bar', 'baz', 'bar', 'bar', 'baz', 'foo', 'bar', 'baz', 'xxx'],
                                  dtype="string"),
                   'i': [pd.Timestamp('2024-01-02 09:15:00'), pd.Timestamp('2024-01-02 09:15:00'),
                         pd.Timestamp('2024-01-02 09:15:00'), pd.Timestamp('2024-01-02 15:45:00'),
                         pd.Timestamp('2024-01-03 07:00:00'), pd.Timestamp('2024-01-03 14:00:00'),
                         pd.Timestamp('2024-01-04 10:00:00'), pd.Timestamp('2024-01-04 18:30:00'),
                         pd.Timestamp('2024-01-05 08:45:00'), pd.Timestamp('2024-01-05 20:15:00')],
                   'j': pd.date_range(start='2024-01-01', periods=10, freq='D'),
                   'k': [pd.Timedelta(hours=0), pd.Timedelta(hours=1), pd.Timedelta(hours=2), pd.Timedelta(hours=3),
                         pd.Timedelta(hours=4), pd.Timedelta(hours=5), pd.Timedelta(hours=6), pd.Timedelta(hours=7),
                         pd.Timedelta(hours=8), pd.Timedelta(hours=9)]
                   })

df = pd.read_csv("penguins.csv")
df['sex'] = df['sex'].astype('category')

print(BestClassification(df, 'sex', 42543, 'crossvalidation', 5, 0.2).accuracy)
