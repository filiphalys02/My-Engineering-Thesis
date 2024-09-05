from datamining._errors import _validate_argument_types1
import pandas as pd
import numpy as np
import math


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
def log_transformation(df: pd.DataFrame, columns: list = None, a: float = 1, b: float = math.e, c: float = 0) \
        -> pd.DataFrame:
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
            df_copy[column] = (df_copy[column] ** alpha - 1) / alpha
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
        df_copy[column] = df_copy[column] ** (1 / root)
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
    :param values: list -> Two-element list
                   The first element will be assigned to the value below the border,
                   The second element will be assigned to the value equal to or higher than the border
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


@_validate_argument_types1
def one_hot_encoding(df: pd.DataFrame, columns: list = None, values: list = [0, 1], prefix_sep: str = '',
                     drop: bool = True):
    """
    The function performs one-hot encoding on selected categorical columns.
    :param df: pandas DataFrame -> Input Data Frame
    :param columns: list -> List of column names to be encoded
                    None -> All categorical columns will be encoded
    :param values: list -> List of two values to replace 1 and 0 in the encoded columns
    :param prefix_sep: str -> Separator between the prefix (column name) and the category name.
    :param drop: bool -> drop or do not drop the original columns after encoding.
    :return: pandas DataFrame -> DataFrame with one-hot encoded columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'string', 'category', 'boolean']).columns.tolist()
    else:
        for column in columns:
            if column not in df.columns:
                raise ValueError(f"There is no column named '{column}' in your DataFrame.")
            if (not pd.api.types.is_bool_dtype(df[column]) and not pd.api.types.is_object_dtype(df[column])
                    and not pd.api.types.is_categorical_dtype(df[column])
                    and not pd.api.types.is_string_dtype(df[column])):
                raise ValueError(f"Column '{column}' is not categorical or boolean.")

    if len(values) != 2:
        raise ValueError(f"Argument 'values' must be a list of 2 elements, not {len(values)}.")

    yes = values[0]
    no = values[1]

    prefixes = {col: col for col in columns}

    df_enc = df.copy()
    for col in columns:
        dummies = pd.get_dummies(df_enc[col], prefix=prefixes.get(col, col), prefix_sep=prefix_sep)
        df_enc = pd.concat([df_enc, dummies], axis=1)
        if drop:
            df_enc = df_enc.drop(columns=[col])

    df_enc = df_enc.map(lambda x: 1 if x is True else (0 if x is False else x))

    df_enc = df_enc.replace({1: yes, 0: no})

    return df_enc
