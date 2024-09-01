from datamining._errors import _validate_argument_types2, _validate_argument_types1
import pandas as pd


@_validate_argument_types2
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
        else:
            raise TypeError("strategy argument must be: ['mean', 'median', 'drop', None, <int>, <float>, <complex>]")


@_validate_argument_types1
def standarization(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    The function performs standardization (Z-score Normalization) of numerical data according to the formula:
            (sample - population mean) / population standard deviation
    :param df: pandas DataFrame -> Input Data Frame
    :param columns: list -> List of column names to standardize
                    None -> All columns will be standardized
    :return: pandas DataFrame -> Input DataFrame with standardized relevant columns
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=['number']).columns.tolist()
    else:
        for element in columns:
            if element not in df.columns:
                raise ValueError(f"There is not a column named '{element}' in your Data Frame.")
            if not pd.api.types.is_numeric_dtype(df[element]):
                raise ValueError(f"Column '{element}' is not numeric.")

    for column in columns:
        mean = df_copy[column].mean()
        std = df_copy[column].std()
        df_copy[column] = (df_copy[column] - mean) / std
        df[column] = df_copy[column]

    return df


@_validate_argument_types1
def normalization_min_max(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    The function performs min-max normalization of numerical data according to the formula:
            (sample - min value) / (max value - min value)
    :param df: pandas DataFrame -> Input Data Frame
    :param columns: list -> List of column names to normalize
                    None -> All columns will be normalized
    :return: pandas DataFrame -> Input DataFrame with normalized relevant columns
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=['number']).columns.tolist()
    else:
        for element in columns:
            if element not in df.columns:
                raise ValueError(f"There is not a column named '{element}' in your Data Frame.")
            if not pd.api.types.is_numeric_dtype(df[element]):
                raise ValueError(f"Column '{element}' is not numeric.")

    for column in columns:
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
        df[column] = df_copy[column]

    return df
