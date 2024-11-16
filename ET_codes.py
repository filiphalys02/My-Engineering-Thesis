import pandas as pd

from datamining._errors import _validate_method_argument_types, _validate_class_argument_types
from datamining.check import check_numeric_data, check_category_data, check_time_series_data, check_time_interval_data, \
    check_data

ramka_testowa = pd.DataFrame({'a': [3, 4, 5, 6, 3, 3, -3, 3, 3, 32],
                              'b': [2.1, 23, 2.1, 2.1, 2.1, -88.2, 2.23, -8.99, 2.2, 2.23],
                              'c': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              'd': [1.3, 2.9, 3.1, 3.5, 5, 6, 7, 8, 11, 9.9],
                              'e': [2, 3, 4, 3, 5, 6, 7, 8, 9, 10],
                              'f': [True, False, False, False, True, True, False, True, True, True],
                              'g': pd.Categorical(['x', 'z', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 's']),
                              'h': pd.Categorical([1, 2, 2, 2, 2, 2, 1, 1, 3, 3]),
                              'i': pd.Series(['foo', 'bar', 'baz', 'bar', 'bar', 'baz', 'foo', 'bar', 'baz', 'xxx'],
                                             dtype="string"),
                              'j': [pd.Timestamp('2024-01-02 09:15:00'), pd.Timestamp('2024-01-02 09:15:00'),
                                    pd.Timestamp('2024-01-02 09:15:00'), pd.Timestamp('2024-01-02 15:45:00'),
                                    pd.Timestamp('2024-01-03 07:00:00'), pd.Timestamp('2024-01-03 14:00:00'),
                                    pd.Timestamp('2024-01-04 10:00:00'), pd.Timestamp('2024-01-04 18:30:00'),
                                    pd.Timestamp('2024-01-05 08:45:00'), pd.Timestamp('2024-01-05 20:15:00')],
                              'k': pd.date_range(start='2024-01-01', periods=10, freq='D'),
                              'l': [pd.Timedelta(hours=0), pd.Timedelta(hours=1), pd.Timedelta(hours=2),
                                    pd.Timedelta(hours=3),
                                    pd.Timedelta(hours=4), pd.Timedelta(hours=5), pd.Timedelta(hours=6),
                                    pd.Timedelta(hours=7),
                                    pd.Timedelta(hours=8), pd.Timedelta(hours=9)]
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
'''
print(check_category_data(df=ramka_testowa, use=False, cat_dist=True))
