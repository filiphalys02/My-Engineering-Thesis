from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
from datamining._errors import _validate_inputs


def _count_line_formula(coefficient, intercept):
    if intercept >= 0:
        formula = f"y = {round(coefficient, 4)}x + {round(intercept, 4)}"
    else:
        formula = f"y = {round(coefficient, 4)}x - {-round(intercept, 4)}"
    return formula


class BestSimpleRegression:
    def __init__(self, df: pd.DataFrame, response: str, divide_method: str = "train_test"):
        self._df = df
        self._response = response
        self._divide_method = divide_method

        self._validate_inputs()
        self._find_best_feature()

    def _validate_inputs(self):
        dic = {
            "df": ["DataFrame", self._df],
            "response": ["str", self._response],
            "divide_data": ["NoneType or str", self._divide_method]
        }
        _validate_inputs(dic)

    def _find_best_feature(self):
        """ Finds the best feature that explains the response variable using linear regression """

        try:
            self._df = self._df.select_dtypes(include=['number'])
            self._df = self._df.select_dtypes(exclude=['timedelta'])
            features = list(self._df.columns)
            features.remove(self._response)
        except ValueError:
            raise ValueError(f"There is no '{self._response}' numeric column in your Data Frame")

        y = self._df[self._response].values
        self._best_feature_dic = {"r-squared": -1}

        for feature in features:
            x = self._df[feature].values
            x = x.reshape(-1, 1)

            r2, mse, rmse, coefficient, intercept = self._choose_divide_method(x, y)

            if r2 > self._best_feature_dic["r-squared"]:
                self._best_feature_dic["r-squared"] = r2
                self._best_feature_dic["feature"] = feature
                self._best_feature_dic["mse"] = mse
                self._best_feature_dic["rmse"] = rmse
                self._best_feature_dic["coefficient"] = coefficient
                self._best_feature_dic["intercept"] = intercept
                self._best_feature_dic["formula"] = _count_line_formula(coefficient, intercept)

        self.best_feature = self._best_feature_dic["feature"]
        self.r_squared = self._best_feature_dic["r-squared"]
        self.mse = self._best_feature_dic["mse"]
        self.rmse = self._best_feature_dic["rmse"]
        self.coefficient = self._best_feature_dic["coefficient"]
        self.intercept = self._best_feature_dic["intercept"]
        self.formula = self._best_feature_dic["formula"]

    def _choose_divide_method(self, x, y):
        if self._divide_method == 'train_test':
            return self._train_test_method(x, y)
        elif self._divide_method == 'crossvalidation':
            return self._crossvalidation_method(x, y)
        else:
            raise ValueError(f"Argument 'divide_method' must be 'train_test' or 'crossvalidation'")

    def _train_test_method(self, x, y):
        """ Train-test split method """
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        LinReg = LinearRegression()
        LinReg.fit(x_train, y_train)
        y_pred = LinReg.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        coefficient = LinReg.coef_
        intercept = LinReg.intercept_

        return r2, mse, rmse, coefficient, intercept

    def _crossvalidation_method(self, x, y):
        """ Cross-validation method """
        kf = KFold(n_splits=5, shuffle=True)
        LinReg = LinearRegression()

        r2s = []
        mses = []
        rmses = []
        coefficients = []
        intercepts = []

        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            LinReg.fit(x_train, y_train)
            y_pred = LinReg.predict(x_test)

            r2s.append(r2_score(y_test, y_pred))
            mse = mean_squared_error(y_test, y_pred)
            mses.append(mse)
            rmses.append(np.sqrt(mse))
            coefficients.append(LinReg.coef_)
            intercepts.append(LinReg.intercept_)

        r2_mean = np.mean(r2s)
        mse_mean = np.mean(mses)
        rmse_mean = np.mean(rmses)
        coefficient_mean = np.mean(coefficients)
        intercept_mean = np.mean(intercepts)

        return r2_mean, mse_mean, rmse_mean, coefficient_mean, intercept_mean

    def plot_model(self):
        """
        Plots the best feature against the response variable with the linear regression model.
        """
        plt.scatter(self._df[self.best_feature], self._df[self._response], label="Data")
        x_range = np.linspace(self._df[self.best_feature].min(), self._df[self.best_feature].max(), 100).reshape(-1, 1)
        y_range_pred = self.coefficient * x_range + self.intercept
        plt.plot(x_range, y_range_pred, color="red", label="Regression Line")
        plt.xlabel(f"Feature: {self.best_feature}")
        plt.ylabel(f"Target: {self._response}")
        plt.title(f"{str(_count_line_formula(self.coefficient, self.intercept))}")
        plt.legend()
        plt.show()
        return self.best_feature
