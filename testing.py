import pandas as pd
from datamining.preprocessing import handle_NaN, standarization, normalization_min_max, log_transformation
from datamining.check import check_numeric_data, check_category_data, check_time_series_data, check_time_interval_data, check_data


df = pd.DataFrame({'a': [None, None, 3, None, 3 , 3, 3, 3, 3, 3],
                   'b': [2.1, 2.1, 2.1, 2.1, 2.1, -88.2, 2.23, -8.99, 2.2, 2.23],
                   'c': ["a", "b", "a", "b","a", "b","a", "b","a", "b"],
                   'd': pd.Categorical(['x', None, 'z', 'x', 'y', 'z', 'x', 'y', 'z', 's']),
                   'e': pd.Series(['foo', 'bar', 'baz', 'foo', 'bar', 'baz', 'foo', 'bar', 'baz', 'xxx'], dtype="string"),
                   'f': [pd.Timestamp('2024-01-01 08:00:00'), pd.Timestamp('2024-01-02 09:15:00'), pd.Timestamp('2024-01-02 09:15:00'), pd.Timestamp('2024-01-02 15:45:00'), pd.Timestamp('2024-01-03 07:00:00'), pd.Timestamp('2024-01-03 14:00:00'), pd.Timestamp('2024-01-04 10:00:00'), pd.Timestamp('2024-01-04 18:30:00'), pd.Timestamp('2024-01-05 08:45:00'), pd.Timestamp('2024-01-05 20:15:00')],
                   'g': pd.date_range(start='2024-01-01', periods=10, freq='D'),
                   'h': [True, True, False, True, False, True, True, False, False, False]
                   })

# check_data(df, 2)


df = pd.DataFrame({'a': [16, 12, 3, 34, 3 , 3, 3, 3, 3, 3],
                   'b': [1, 2, 3, None, 5, 6, 7, 8, 199, 9],
                   'c': [34, 54, None, -6, -99, 45, 56, 4, 5, 1],
                   'd': ['a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e'],
                   'g': pd.date_range(start='2024-01-01', periods=10, freq='D')
                   })

print(log_transformation(df, ['a', 'c'], a=2, b=2, c=99.5))
