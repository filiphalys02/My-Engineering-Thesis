from datamining._errors import _validate_argument_types2, _validate_argument_types1
import pandas as pd
import math
import numpy as np


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


@_validate_argument_types1
def log_transformation(df: pd.DataFrame, columns: list = None, a: float = 1, b: float = math.e, c: float = 0) -> pd.DataFrame:
    """
    The function performs logarithm transformation of numerical data according to the formula:
            a * log_b(sample + c)
    :param df: pandas DataFrame -> Input Data Frame
    :param columns: list -> List of column names to transform
                    None -> All numerical columns will be transformed
    :param a: float -> Scaling factor
    :param b: float -> Base of the logarithm
    :param c: float -> Offset to be added to each value before taking the logarithm
    :return: pandas DataFrame -> Input DataFrame with transformed relevant columns
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=['number']).columns.tolist()
    else:
        for element in columns:
            if element not in df.columns:
                raise ValueError(f"There is no column named '{element}' in your DataFrame.")
            if not pd.api.types.is_numeric_dtype(df[element]):
                raise ValueError(f"Column '{element}' is not numeric.")

    for column in columns:
        values_with_c = df_copy[column] + c
        if (values_with_c <= 0).any():
            raise ValueError(f"Cannot count the logarithm of {column} column. Check your offset {c}.")

        df_copy[column] = a * np.log(values_with_c) / np.log(b)
    return df_copy


@_validate_argument_types1
def normalization_box_kox(df: pd.DataFrame, columns: list = None, alpha: int = 1) -> pd.DataFrame:
    """
    The function performs box-kox normalization of numerical data according to the formula:
            (sample^alpha - 1) / alpha, if alpha is not equal to 0
            ln(sample), if alpha is equal to 0
    :param alpha: int -> alpha
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

    if alpha != 0:
        for column in columns:
            df_copy[column] = (df_copy[column]**alpha - 1) / alpha
    else:
        for column in columns:
            df_copy[column] = np.log(df_copy[column])

    df[column] = df_copy[column]

    return df


@_validate_argument_types1
def root_transformation(df: pd.DataFrame, columns: list = None, root: int = 1) -> pd.DataFrame:
    """
    The function performs root transformation of numerical data according to the formula:
            sample ^ (1/root)
    :param root: int -> root
    :param df: pandas DataFrame -> Input Data Frame
    :param columns: list -> List of column names to transform
                    None -> All columns will be transformed
    :return: pandas DataFrame -> Input DataFrame with transformed relevant columns
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
        df_copy[column] = df_copy[column] ** (1/root)
        df[column] = df_copy[column]

    return df


@_validate_argument_types1
def root_transformation(df: pd.DataFrame, columns: list = None, root: int = 1) -> pd.DataFrame:
    """
    The function performs root transformation of numerical data according to the formula:
            sample ^ (1/root)
    :param root: int -> root
    :param df: pandas DataFrame -> Input Data Frame
    :param columns: list -> List of column names to transform
                    None -> All columns will be transformed
    :return: pandas DataFrame -> Input DataFrame with transformed relevant columns
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
        df_copy[column] = df_copy[column] ** (1/root)
        df[column] = df_copy[column]

    return df


@_validate_argument_types1
def binarization(df: pd.DataFrame, columns: list = None, border: float = 0, values: list = [0, 1]) -> pd.DataFrame:
    """
    The function performs binarization of numerical data
    :param df: pandas DataFrame -> Input Data Frame
    :param columns: list -> List of column names to binarize
                    None -> All columns will be binarized
    :param border: float -> The boundary against which binarization will be performed
    :param values: Two-element list,
                   the first element will be assigned to the value below the border,
                   the second element will be assigned to the value equal to or higher than the border
    :return: pandas DataFrame -> Input DataFrame with binarized relevant columns
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

    if len(values) != 2:
        raise ValueError(f"Argument 'list' must be a list of 2 elements, not {len(values)}.")

    for column in columns:
        low = values[0]
        high = values[1]
        df_copy[column] = df_copy[column].apply(lambda x: low if x < border else (high if x >= border else np.nan))

        df[column] = df_copy[column]

    return df