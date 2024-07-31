import pandas as pd
from datamining.preprocessing import handle_NaN
from datamining.check import check_numeric_data
import pandas

df = pd.DataFrame({'a': [None, None, 3, None, 3, 3, 3, 3, 3, 3c],
                   'b': [2, 2, 3, 1, 2, 3, 1, 2, 1, 3]})

# print(handle_NaN(df, strategy=12))

print(check_numeric_data(df, 2))
print(df.describe())
