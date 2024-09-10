from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from datamining._errors import _validate_argument_types1, _validate_inputs
import pandas as pd


class BestSimpleRegression:
    def __init__(self, df: pd.DataFrame, response: str):
        self._df = df
        self._response = response

        self._validate_inputs()
        self._find_best_feature()

    def _validate_inputs(self):
        dic = {
            "df": ["DataFrame", self._df]
        }
        _validate_inputs(dic)

    def _find_best_feature(self):
        """ Finds the best feature that explains the response variable using linear regression """

        try:
            self._df = self._df.select_dtypes(include=['number'])
            self._df = self._df.select_dtypes(exclude=['timedelta', 'datetime'])
            features = list(self._df.columns)
            features.remove(self._response)
        except ValueError:
            raise ValueError(f"There is not '{self._response}' numeric column in your Data Frame")

        y = self._df[self._response].values
        self._best_feature_dic = {"r-squared": -1}

        for feature in features:
            x = self._df[feature].values
            x = x.reshape(-1, 1)

            self._train_test_for_find_best_feature(feature, x, y)
            # print(f"Evaluated feature: {feature}, Current best feature dict: {self._best_feature_dic}")

        self.best_feature = self._best_feature_dic["feature"]
        # print(f"Best feature: {self.best_feature}, R-squared: {self._best_feature_dic['r-squared']}")

    def _train_test_for_find_best_feature(self, feature, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        LinReg = LinearRegression()
        LinReg.fit(x_train, y_train)
        y_pred = LinReg.predict(x_test)

        r2 = round(r2_score(y_test, y_pred), 4)

        if r2 > self._best_feature_dic["r-squared"]:
            self._best_feature_dic["r-squared"] = r2
            self._best_feature_dic["feature"] = feature

