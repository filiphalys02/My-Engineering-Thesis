import pandas as pd
from datamining._errors import _validate_argument_types


@_validate_argument_types
def check_numeric_data(df: pd.DataFrame, round: int = 1, use: bool = False):
    """
    The function summarizes numerical columns in a data frame using statistical measures.
    :param df: pandas DataFrame -> Input Data Frame
    :param round: int -> The number of decimal places to which the function will round the measures.
    :param use: boolean -> To use the output as a pandas Data Frame, set to True.
    :return: str -> If param use is False
             pandas DataFrame -> If param use is True

            Output measures:
            NAME -> Column name
            TYPE -> Data type
            NaN -> Number of missing values
            AVG -> Mean
            Q25 -> 25th percentile
            Q50 -> 50th percentile / median
            Q75 -> 75th percentile
            IQR -> Interquartile range
            MIN -> Minimum value
            MAX -> Maximum value
            STD -> Standard deviation
            SUM -> Sum of values
            LOW OUT -> Number of lower outliers
            UPP OUT -> Number of upper outliers
    """

    df = df.select_dtypes(include=['number'])

    result_df = pd.DataFrame()

    names = list()
    types = list()
    nans = list()
    means = list()
    q25s = list()
    q50s = list()
    q75s = list()
    iqrs = list()
    mins = list()
    maxs = list()
    stds = list()
    sums = list()
    lout = list()
    uout = list()

    for column in df.columns:
        names.append(column)
        types.append(df[column].dtype)
        nans.append(df[column].isna().sum())
        means.append(df[column].mean())
        q25s.append(df[column].quantile(0.25))
        q50s.append(df[column].quantile(0.5))
        q75s.append(df[column].quantile(0.75))
        iqrs.append(_count_iqr(df, column))
        mins.append(df[column].min())
        maxs.append(df[column].max())
        stds.append(df[column].std())
        sums.append(df[column].sum())
        lout.append((df[column] < (df[column].quantile(0.25) - 1.5 * _count_iqr(df, column))).sum())
        uout.append((df[column] > (df[column].quantile(0.75) + 1.5 * _count_iqr(df, column))).sum())

    result_df["NAME"] = names
    result_df["TYPE"] = types
    result_df["NaN"] = nans
    result_df["AVG"] = [f"{mean:.{round}f}" for mean in means]
    result_df['Q25'] = [f"{q25:.{round}f}" for q25 in q25s]
    result_df['Q50'] = [f"{q50:.{round}f}" for q50 in q50s]
    result_df['Q75'] = [f"{q75:.{round}f}" for q75 in q75s]
    result_df["IQR"] = [f"{iqr:.{round}f}" for iqr in iqrs]
    result_df["MIN"] = [f"{min:.{round}f}" for min in mins]
    result_df["MAX"] = [f"{max:.{round}f}" for max in maxs]
    result_df["STD"] = [f"{std:.{round}f}" for std in stds]
    result_df["SUM"] = [f"{sum:.{round}f}" for sum in sums]
    result_df["LOW OUT"] = lout
    result_df["UPP OUT"] = uout

    if use is True:
        return result_df
    if use is False:
        return result_df.to_string(index=False)


def _count_iqr(df, column):
    return df[column].quantile(0.75) - df[column].quantile(0.25)
