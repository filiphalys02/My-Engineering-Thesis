import pandas as pd
from datamining.preprocessing import handle_NaN
import pandas

df = pd.DataFrame({'a': [None, None, 3, None, 5, 6, 7, 8, 9, 1],
                   'b': [None, 2, 3, 1, 2, 3, 1, 2, None, 3]})

print(handle_NaN(df, strategy=12))
