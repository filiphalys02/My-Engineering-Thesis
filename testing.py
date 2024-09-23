import numpy as np
import pandas as pd
from datamining.preprocessing import handle_numeric_NaN, handle_category_NaN
from datamining.normalizations import (standarization, normalization_min_max, log_transformation,
                                      normalization_box_kox, root_transformation, binarization, one_hot_encoding)
from datamining.check import (check_numeric_data, check_category_data, check_time_series_data, check_time_interval_data,
                              check_data)
from datamining.regression import BestSimpleLinearRegression, BestMultipleLinearRegression

pd.set_option('future.no_silent_downcasting', True)

df = pd.DataFrame({'a': [3, 4, 5, 6, 3 , 3, -3, 3, 3, 32],
                   'b': [2.1, 23, 2.1, 2.1, 2.1, -88.2, 2.23, -8.99, 2.2, 2.23],
                   'j': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'k': [2, 3, 3, 3, 5, 6, 7, 8, 9, 9],
                   'l': [2, 3, 4, 3, 5, 6, 7, 8, 9, 10],
                   'c': ["a", "b", "a", "b", "None", "b", "a", "b", "a", "b"],
                   'd': pd.Categorical(['x', 'z', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 's']),
                   'e': pd.Series(['foo', 'bar', 'baz', 'bar', 'bar', 'baz', 'foo', 'bar', 'baz', 'xxx'], dtype="string"),
                   'f': [pd.Timestamp('2024-01-02 09:15:00'), pd.Timestamp('2024-01-02 09:15:00'), pd.Timestamp('2024-01-02 09:15:00'), pd.Timestamp('2024-01-02 15:45:00'), pd.Timestamp('2024-01-03 07:00:00'), pd.Timestamp('2024-01-03 14:00:00'), pd.Timestamp('2024-01-04 10:00:00'), pd.Timestamp('2024-01-04 18:30:00'), pd.Timestamp('2024-01-05 08:45:00'), pd.Timestamp('2024-01-05 20:15:00')],
                   'g': pd.date_range(start='2024-01-01', periods=10, freq='D'),
                   'i': [pd.Timedelta(hours=0), pd.Timedelta(hours=1), pd.Timedelta(hours=2), pd.Timedelta(hours=3), pd.Timedelta(hours=4), pd.Timedelta(hours=5), pd.Timedelta(hours=6), pd.Timedelta(hours=7), pd.Timedelta(hours=8), pd.Timedelta(hours=9)]
                   })

print(BestMultipleLinearRegression(df, 'k', 122, 'train_test').r_squared)

