from datamining._errors import _validate_argument_types2, _validate_argument_types1
import pandas as pd


@_validate_argument_types1
def handle_numeric_NaN(df: pd.DataFrame, columns: list = None, strategy=None):
    """
    Function helps to handle missing values in your Data Frame in numeric columns.
    :param df: pandas DataFrame -> Input DataFrame
    :param columns: list -> List of column names
                   None -> Every numeric column will be included
    :param strategy: numeric -> Replace missing values with numeric value
                     'mean' -> Replace missing values with column mean
                     'median' -> Replace missing values with column median
                     'min' -> Replace missing values with minimum value
                     'max' -> Replace missing values with maximum value
                     'drop' -> Drop rows with missing values
    :return: pandas DataFrame -> Output Data Frame
    """
    if columns is None:
        columns = df.columns
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"'{column}' is not a column in your DataFrame")
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if strategy == 'mean':
            df[column] = df[column].fillna(df[column].mean())
        elif strategy == 'median':
            df[column] = df[column].fillna(df[column].median())
        elif strategy == 'min':
            df[column] = df[column].fillna(df[column].min())
        elif strategy == 'max':
            df[column] = df[column].fillna(df[column].max())
        elif strategy == 'drop':
            df = df.dropna(subset=[column])
        elif isinstance(strategy, (int, float, complex)):
            df[column] = df[column].fillna(strategy)
        elif strategy is None:
            df[column] = df[column].fillna(0)
        else:
            raise (TypeError("strategy argument must be one of: "
                             "['mean', 'median', 'drop', None, <int>, <float>, <complex>]"))
    return df


