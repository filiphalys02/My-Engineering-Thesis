import pandas as pd
from datamining.preprocessing import handle_NaN
from datamining.check import check_numeric_data, check_category_data
import pandas

df = pd.DataFrame({'a': [None, None, 3, None, 3 , 3, 3, 3, 3, 3],
                   'b': [2.1, 2.1, 2.1, 2.1, 2.1, -88.2, 2.23, -8.99, 2.2, 2.23],
                   'c': ["a", "b", "a", "b","a", "b","a", "b","a", "b"],
                   'd': pd.Categorical(['x', None, 'z', 'x', 'y', 'z', 'x', 'y', 'z', 's']),
                   'e': pd.Series(['foo', 'bar', 'baz', 'foo', 'bar', 'baz', 'foo', 'bar', 'baz', 'xxx'], dtype="string"),
                   })

print(check_numeric_data(df, 2))

print(check_category_data(df))
print(type(check_category_data(df, True)))
