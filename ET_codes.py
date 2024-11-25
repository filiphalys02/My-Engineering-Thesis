import pandas as pd

from datamining._errors import _validate_method_argument_types, _validate_class_argument_types
from datamining.check import check_numeric_data, check_category_data, check_time_series_data, check_time_interval_data, check_data
from datamining.transformations import standarization, normalization_min_max, log_transformation, transformation_box_kox, root_transformation, binarization
from datamining.regression import BestSimpleLinearRegression, BestMultipleLinearRegression

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
                              'l': [pd.Timedelta(hours=0, minutes=5, seconds=30),
                                    pd.Timedelta(hours=1, minutes=10, seconds=15),
                                    pd.Timedelta(hours=2, minutes=20, seconds=45),
                                    pd.Timedelta(hours=3, minutes=15, seconds=0),
                                    pd.Timedelta(hours=4, minutes=25, seconds=10),
                                    pd.Timedelta(hours=5, minutes=30, seconds=30),
                                    pd.Timedelta(hours=6, minutes=45, seconds=50),
                                    pd.Timedelta(hours=7, minutes=50, seconds=20),
                                    pd.Timedelta(hours=8, minutes=5, seconds=5),
                                    pd.Timedelta(hours=9, minutes=15, seconds=35)],
                              'm': [pd.Timedelta(days=5),
                                    pd.Timedelta(days=10),
                                    pd.Timedelta(days=15),
                                    None,
                                    pd.Timedelta(days=25),
                                    pd.Timedelta(days=30),
                                    pd.Timedelta(days=35),
                                    pd.Timedelta(days=40),
                                    pd.Timedelta(days=45),
                                    pd.Timedelta(days=45)]
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
standarization(df)
normalization_min_max(df, ['a', 'b'])
print(log_transformation(df, ['a', 'b'], 2.0,10.0,1.0))
print(transformation_box_kox(df, columns=['a', 'b'], alpha=3))
print(root_transformation(df, ['a', 'b'], root=2))
print(binarization(df, ['a', 'b'], border=8, values=['<8', '>=8']))
'''

df = pd.DataFrame({
    'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'b': [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
})







#BestMultipleLinearRegression(ramka_testowa, 'd').plot_model()
