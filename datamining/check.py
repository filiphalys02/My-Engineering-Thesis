import pandas as pd
from datamining._errors import _validate_method_argument_types


@_validate_method_argument_types
def check_numeric_data(df: pd.DataFrame, round: int = 1, use: bool = False):
    """
    The function summarizes numerical columns in a data frame using statistical measures.
    :param df: pandas DataFrame -> Input Data Frame
    :param round: int -> The number of decimal places to which the function will round the measures
    :param use: boolean -> To use the output as a pandas Data Frame, set to True
    :return: str -> If param use is False
             pandas DataFrame -> If param use is True

            Output measures:
            NAME -> Column name
            TYPE -> Data type
            MIS -> Number of missing values
            AVG -> Mean
            Q25 -> 25th percentile
            Q50 -> 50th percentile / median
            Q75 -> 75th percentile
            IQR -> Interquartile range
            MIN -> Minimum value
            MAX -> Maximum value
            RAN -> Range
            STD -> Standard deviation
            SUM -> Sum of values
            LOW OUT -> Number of lower outliers
            UPP OUT -> Number of upper outliers

        *** The function will not work for a column with complex numeric variables ***
    """

    df = df.select_dtypes(include=['number'])
    df = df.select_dtypes(exclude=['timedelta'])

    if df.empty:
        return f'There is no numeric columns in Your Data Frame'

    result_df = pd.DataFrame()

    names = list()
    types = list()
    miss = list()
    means = list()
    q25s = list()
    q50s = list()
    q75s = list()
    iqrs = list()
    mins = list()
    maxs = list()
    rans = list()
    stds = list()
    sums = list()
    lout = list()
    uout = list()

    for column in df.columns:
        names.append(column)
        types.append(df[column].dtype)
        miss.append(df[column].isna().sum())
        means.append(df[column].mean())
        q25s.append(df[column].quantile(0.25))
        q50s.append(df[column].quantile(0.5))
        q75s.append(df[column].quantile(0.75))
        iqrs.append(_count_iqr(df, column))
        mins.append(df[column].min())
        maxs.append(df[column].max())
        rans.append(df[column].max()-df[column].min())
        stds.append(df[column].std())
        sums.append(df[column].sum())
        lout.append((df[column] < (df[column].quantile(0.25) - 1.5 * _count_iqr(df, column))).sum())
        uout.append((df[column] > (df[column].quantile(0.75) + 1.5 * _count_iqr(df, column))).sum())

    result_df["NAME"] = names
    result_df["TYPE"] = types
    result_df["MIS"] = miss
    result_df["AVG"] = [f"{mean:.{round}f}" for mean in means]
    result_df["Q25"] = [f"{q25:.{round}f}" for q25 in q25s]
    result_df["MED"] = [f"{q50:.{round}f}" for q50 in q50s]
    result_df["Q75"] = [f"{q75:.{round}f}" for q75 in q75s]
    result_df["IQR"] = [f"{iqr:.{round}f}" for iqr in iqrs]
    result_df["MIN"] = [f"{min:.{round}f}" for min in mins]
    result_df["MAX"] = [f"{max:.{round}f}" for max in maxs]
    result_df["RAN"] = [f"{ran:.{round}f}" for ran in rans]
    result_df["STD"] = [f"{std:.{round}f}" for std in stds]
    result_df["SUM"] = [f"{sum:.{round}f}" for sum in sums]
    result_df["LOW OUT"] = lout
    result_df["UPP OUT"] = uout

    if use is True:
        return result_df
    if use is False:
        return result_df.to_string(index=False)


@_validate_method_argument_types
def check_category_data(df: pd.DataFrame, use: bool = False, cat_dist: bool = False):
    """
    The function summarizes categorical columns in a data frame using statistical measures.
    :param df: pandas DataFrame -> Input Data Frame
    :param use: boolean -> To use the output as a pandas Data Frame, set to True
    :param cat_dist: boolean -> To check category distribution in each column, set to True
    :return: str -> If param use is False
             pandas DataFrame -> If param use is True

            Output measures:
            NAME -> Column name
            TYPE -> Data type
            MIS -> Number of missing values
            UNIQUE -> Number of unique categories
            MODE -> Most frequently occurring category (modal value)
            FREQ -> Frequency of the most frequently occurring category
    """

    df = df.select_dtypes(include=['object', 'string', 'category', 'boolean'])

    if df.empty:
        return f'There is no categorical columns in Your Data Frame'

    result_df = pd.DataFrame()

    names = list()
    types = list()
    miss = list()
    unis = list()
    mods = list()
    fres = list()
    firsts = list()
    lasts = list()

    for column in df.columns:
        names.append(column)
        types.append(df[column].dtype)
        miss.append(df[column].isna().sum())
        unis.append(df[column].nunique())
        mode_series = df[column].mode()
        firsts.append(df[column].iloc[0])
        lasts.append(df[column].iloc[-1])
        if mode_series.empty:
            mods.append(None)
            fres.append(0)
        else:
            mods.append(mode_series[0])
            fres.append(df[column].value_counts().get(mode_series[0], 0))

        if cat_dist:
            result_cd = pd.DataFrame()
            amount_cd = df[column].value_counts()
            percentage_cd = df[column].value_counts(normalize=True) * 100
            rank_cd = amount_cd.rank(ascending=False)
            result_cd["CATEGORY"] = amount_cd.index
            result_cd["AMOUNT"] = amount_cd.values
            result_cd["PERCENTAGE"] = percentage_cd.values
            result_cd["RANK"] = rank_cd.values
            result_cd = result_cd.to_string(index=False)
            print(f"COLUMN: {column}")
            print(result_cd)
            print("\n")

    if cat_dist:
        return ""

    result_df["NAME"] = names
    result_df["TYPE"] = types
    result_df["MIS"] = miss
    result_df["UNIQUE"] = unis
    result_df["MODE"] = mods
    result_df["FREQ"] = fres
    result_df["FIRST"] = firsts
    result_df["LAST"] = lasts

    if use is True:
        return result_df
    if use is False:
        return result_df.to_string(index=False)


@_validate_method_argument_types
def check_time_series_data(df: pd.DataFrame, use: bool = False):
    """
    The function summarizes time series columns in a data frame using statistical measures.
    :param df: pandas DataFrame -> Input Data Frame
    :param use: boolean -> To use the output as a pandas Data Frame, set to True
    :return: str -> If param use is False
             pandas DataFrame -> If param use is True

            Output measures:
            NAME -> Column name
            TYPE -> Data type
            MIS -> Number of missing values
            MIN -> Earliest date
            MAX -> Latest date
            RAN -> Difference between latest date and earliest date
            AVG -> Average date
            MED -> Median date
            STD -> Date standard deviation
            UNIQUE -> Number of unique dates
    """

    df = df.select_dtypes(include=['datetime'])

    if df.empty:
        return f'There is no time series columns in Your Data Frame'

    result_df = pd.DataFrame()

    names = list()
    types = list()
    miss = list()
    mins = list()
    maxs = list()
    rans = list()
    avgs = list()
    meds = list()
    stds = list()
    unis = list()

    for column in df.columns:
        names.append(column)
        types.append(df[column].dtype)
        miss.append(df[column].isna().sum())
        mins.append(df[column].min())
        maxs.append(df[column].max())
        rans.append(df[column].max() - df[column].min())
        avgs.append(df[column].mean())
        meds.append(df[column].quantile(0.5))
        stds.append(df[column].std())
        unis.append(df[column].nunique())

    result_df["NAME"] = names
    result_df["TYPE"] = types
    result_df["MIS"] = miss
    result_df["MIN"] = mins
    result_df["MAX"] = maxs
    result_df["RAN"] = rans
    result_df["AVG"] = avgs
    result_df["MED"] = meds
    result_df["STD"] = stds
    result_df["UNIQUE"] = unis

    if use is True:
        return result_df
    if use is False:
        return result_df.to_string(index=False)


@_validate_method_argument_types
def check_time_interval_data(df: pd.DataFrame, use: bool = False):
    """
    The function summarizes time interval columns in a data frame using statistical measures.
    :param df: pandas DataFrame -> Input Data Frame
    :param use: boolean -> To use the output as a pandas Data Frame, set to True
    :return: str -> If param use is False
             pandas DataFrame -> If param use is True

            Output measures:
            NAME -> Column name
            TYPE -> Data type
            MIS -> Number of missing values
            AVG -> Average interval
            MED -> Median interval
            IQR -> Interquartile range interval
            MIN -> Shortest interval
            MAX -> Longest interval
            RAN -> Difference between longest and shortest interval
            STD -> Interval standard deviation
            SUM -> Total length of all intervals
            UNIQUE -> Number of unique intervals
    """
    df = df.select_dtypes(include=['timedelta'])

    if df.empty:
        return f'There is no time interval columns in Your Data Frame'

    result_df = pd.DataFrame()

    names = list()
    types = list()
    miss = list()
    avgs = list()
    q50s = list()
    iqrs = list()
    mins = list()
    maxs = list()
    rans = list()
    stds = list()
    sums = list()
    unis = list()

    for column in df.columns:
        names.append(column)
        types.append(df[column].dtype)
        miss.append(df[column].isna().sum())
        avgs.append(pd.to_timedelta(round(df[column].mean().total_seconds()), unit='s'))
        q50s.append(pd.to_timedelta(round(df[column].quantile(0.5).total_seconds()), unit='s'))
        iqrs.append(pd.to_timedelta(round(_count_iqr(df, column).total_seconds()), unit='s'))
        mins.append(pd.to_timedelta(round(df[column].min().total_seconds()), unit='s'))
        maxs.append(pd.to_timedelta(round(df[column].max().total_seconds()), unit='s'))
        rans.append(pd.to_timedelta(round((df[column].max() - df[column].min()).total_seconds()), unit='s'))
        stds.append(pd.to_timedelta(round(df[column].std().total_seconds()), unit='s'))
        sums.append(pd.to_timedelta(round((df[column].sum() - df[column].min()).total_seconds()), unit='s'))
        unis.append(df[column].nunique())

    result_df["NAME"] = names
    result_df["TYPE"] = types
    result_df["MIS"] = miss
    result_df["AVG"] = avgs
    result_df["MED"] = q50s
    result_df["IQR"] = iqrs
    result_df["MIN"] = mins
    result_df["MAX"] = maxs
    result_df["RAN"] = rans
    result_df["STD"] = stds
    result_df["SUM"] = sums
    result_df["UNIQUE"] = unis

    if use is True:
        return result_df
    if use is False:
        return result_df.to_string(index=False)


@_validate_method_argument_types
def check_data(df: pd.DataFrame, round: int = 1):
    """
    The function summarizes all columns in a data frame using statistical measures.
    :param df: pandas DataFrame -> Input Data Frame
    :param round: int -> The number of decimal places to which the function will round the measures (numeric data)
    :return: None
    """
    if check_numeric_data(df, round) != 'There is no numeric columns in Your Data Frame':
        print(f'--- NUMERIC DATA --- \n {check_numeric_data(df, round)} \n')
    if check_category_data(df) != 'There is no categorical columns in Your Data Frame':
        print(f'--- CATEGORY DATA --- \n {check_category_data(df)} \n')
    if check_time_series_data(df) != 'There is no time series columns in Your Data Frame':
        print(f'--- TIME SERIES DATA --- \n {check_time_series_data(df)} \n')
    if check_time_interval_data(df) != 'There is no time interval columns in Your Data Frame':
        print(f'--- TIME INTERVAL DATA --- \n {check_time_interval_data(df)} \n')


@_validate_method_argument_types
def _count_iqr(df, column):
    return df[column].quantile(0.75) - df[column].quantile(0.25)
