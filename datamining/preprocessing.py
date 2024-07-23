from datamining._errors import _validate_argument_types
import pandas as pd


@_validate_argument_types
def handle_NaN(df: pd.DataFrame, column: str = None, strategy=None) -> pd.DataFrame:

    if column is not None:

        if column not in df.columns:
            raise ValueError(f'{column} is not a column in your DataFrame')

        if strategy == 'mean':
            df[column] = df[column].fillna(df[column].mean())
            return df
        elif strategy == 'median':
            df[column] = df[column].fillna(df[column].median())
            return df
        elif strategy == 'drop':
            return df.dropna(subset=[column])
        elif isinstance(strategy, (int, float, complex)):
            df[column] = df[column].fillna(strategy)
            return df
        elif strategy is None:
            df[column] = df[column].fillna(0)
            return df
        elif strategy == 'regression':
            print('regression')
        else:
            raise TypeError(
                "strategy argument must be: ['mean', 'median', 'drop', None, <int>, <float>, <complex>]")

    else:

        if strategy == 'mean':
            df = df.fillna(df.mean())
            return df
        elif strategy == 'median':
            df = df.fillna(df.median())
            return df
        elif strategy == 'drop':
            return df.dropna()
        elif isinstance(strategy, (int, float, complex)):
            df = df.fillna(strategy)
            return df
        elif strategy is None:
            df = df.fillna(0)
            return df
        elif strategy == 'regression':
            print('regression')
        else:
            raise TypeError("strategy argument must be: ['mean', 'median', 'drop', None, <int>, <float>, <complex>]")
