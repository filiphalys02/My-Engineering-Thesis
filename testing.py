import pandas as pd
from datamining.preprocessing import handle_NaN
from datamining.check import check_numeric_data
import pandas

df = pd.DataFrame({'a': [None, None, 3, None, 3 , 3, 3, 3, 3, 3],
                   'b': [2.1, 2.1, 2.1, 2.1, 2.1, -88.2, 2.23, -8.99, 2.2, 2.23],
                   'c': ["a", "b", "a", "b","a", "b","a", "b","a", "b",]})

# print(handle_NaN(df, strategy=12))

print(check_numeric_data(df, 2))

