import pandas as pd
from datamining._errors import _validate_argument_types


@_validate_argument_types
def check_numeric_data(df: pd.DataFrame, round: int = 1):
    result_df = pd.DataFrame()

    names = list()
    types = list()
    nans = list()
    means = list()
    q25s = list()
    q50s = list()
    q75s = list()
    mins = list()
    maxs = list()
    stds = list()
    sums = list()

    for column in df.columns:
        names.append(column)
        types.append(df[column].dtype)
        nans.append(df[column].isna().sum())
        means.append(df[column].mean())
        q25s.append(df[column].quantile(0.25))
        q50s.append(df[column].quantile(0.5))
        q75s.append(df[column].quantile(0.75))
        mins.append(df[column].min())
        maxs.append(df[column].max())
        stds.append(df[column].std())
        sums.append(df[column].sum())

    result_df["NAME"] = names
    result_df["TYPE"] = types
    result_df["NaN"] = nans
    result_df["AVG"] = [f"{mean:.{round}f}" for mean in means]
    result_df['Q25'] = [f"{q25:.{round}f}" for q25 in q25s]
    result_df['Q50'] = [f"{q50:.{round}f}" for q50 in q50s]
    result_df['Q75'] = [f"{q75:.{round}f}" for q75 in q75s]
    result_df["MIN"] = [f"{min:.{round}f}" for min in mins]
    result_df["MAX"] = [f"{max:.{round}f}" for max in maxs]
    result_df["STD"] = [f"{std:.{round}f}" for std in stds]
    result_df["SUM"] = [f"{sum:.{round}f}" for sum in sums]

    return result_df.to_string(index=False)


