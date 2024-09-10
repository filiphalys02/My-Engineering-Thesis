from datamining._errors import _validate_argument_types1, _validate_inputs
import pandas as pd


class BestSimpleRegression:
    def __init__(self, df: pd.DataFrame):
        self._df = df

        self._validate_inputs()

    def _validate_inputs(self):
        dic = {
            "df": ["DataFrame", self._df]
        }
        _validate_inputs(dic)
