from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import itertools
import pandas as pd
import numpy as np
from datamining._errors import _validate_class_argument_types


def _count_line_formula(coefficient, intercept, response, best_feature):
    if intercept >= 0:
        formula = f"{response} = {round(coefficient, 4)}*{best_feature} + {round(intercept, 4)}"
    else:
        formula = f"{response} = {round(coefficient, 4)}*{best_feature} - {-round(intercept, 4)}"
    return formula


def _count_line_formula_2(coefficients, intercept, response, features):
    elements = []
    for coef, feature in zip(coefficients, features):
        if coef < 0:
            elements.append(f"- {-coef:.4f}*{feature}")
        else:
            elements.append(f"+ {coef:.4f}*{feature}")

    formula = f"{response} = {intercept:.4f} " + " ".join(elements)
    if formula.startswith(f"{response} = +"):
        formula = formula.replace(f"{response} = +", f"{response} =")

    return formula


def _generate_independent_variables_combinations_list(list_of_independent_variables):
    list_of_all_combinations = []
    for r in range(2, len(list_of_independent_variables) + 1):
        combinations = list(itertools.combinations(list_of_independent_variables, r))
        list_of_all_combinations.extend(combinations)
    return list_of_all_combinations


class BestSimpleLinearRegression:
    def __init__(self, df: pd.DataFrame, response: str, set_seed: int = None, divide_method: str = "train_test",
                 k: int = 5, test_size: float = 0.2):
        """
        The objects are models of simple linear regression for the dependent variable and the independent variable
        that best fits it, using the r-squared coefficient.
        :param df: pandas DataFrame -> Input data frame
        :param response: string -> Name of dependent variable
        :param set_seed: int -> Seed number for reproducibility
        :param divide_method: 'train_test' -> Split data with train and test method
                              'crossvalidation' -> Split data with crossvalidation method
        :param k: int -> Folds in crossvalidation
        :param: test_size: float -> Size of the test set
        """
        self._df = df
        self._response = response
        self._set_seed = set_seed
        self._divide_method = divide_method
        self._k = k
        self._test_size = test_size

        self._validate_inputs()
        self._find_best_feature()

    def _validate_inputs(self):
        dic = {
            "df": ["DataFrame", self._df],
            "response": ["str", self._response],
            "set_seed": ["NoneType or int", self._set_seed],
            "divide_data": ["NoneType or str", self._divide_method],
            "k": ["int", self._k],
            "test_size": ["float", self._test_size]
        }
        _validate_class_argument_types(dic)

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

            r2, mse, rmse, coefficient, intercept, rss, rmspe, mape, model, y_pred = self._choose_divide_method(x, y)

            if r2 > self._best_feature_dic["r-squared"]:
                self._best_feature_dic["r-squared"] = r2
                self._best_feature_dic["feature"] = feature
                self._best_feature_dic["mse"] = mse
                self._best_feature_dic["rmse"] = rmse
                self._best_feature_dic["coefficient"] = coefficient
                self._best_feature_dic["intercept"] = intercept
                self._best_feature_dic["formula"] = _count_line_formula(coefficient, intercept, self._response, feature)
                self._best_feature_dic["rss"] = rss
                self._best_feature_dic["rmspe"] = rmspe
                self._best_feature_dic["mape"] = mape
                self._best_feature_dic["model"] = model
                self._best_feature_dic["y_pred"] = y_pred

        self.best_feature = self._best_feature_dic["feature"]
        self.r_squared = self._best_feature_dic["r-squared"]
        self.mse = self._best_feature_dic["mse"]
        self.rmse = self._best_feature_dic["rmse"]
        self.coefficient = self._best_feature_dic["coefficient"]
        self.intercept = self._best_feature_dic["intercept"]
        self.formula = self._best_feature_dic["formula"]
        self.rss = self._best_feature_dic["rss"]
        self.rmspe = self._best_feature_dic["rmspe"]
        self.mape = self._best_feature_dic["mape"]
        self.model = self._best_feature_dic["model"]
        self.y_pred = self._best_feature_dic["y_pred"]

    def _choose_divide_method(self, x, y):
        if self._divide_method == 'train_test':
            return self._train_test_method(x, y)
        elif self._divide_method == 'crossvalidation':
            return self._crossvalidation_method(x, y)
        else:
            raise ValueError(f"Argument 'divide_method' must be 'train_test' or 'crossvalidation'")

    def _train_test_method(self, x, y):
        """ Train-test split method """
        if self._set_seed is None:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self._test_size)
        else:
            if 0 <= self._set_seed <= 4294967295:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self._test_size,
                                                                    random_state=self._set_seed)
            else:
                raise ValueError(f"Argument 'set_seed' must be in the range [0, 4294967295]")

        LinReg = LinearRegression()
        LinReg.fit(x_train, y_train)
        y_pred = LinReg.predict(x_test)

        r2 = r2_score(y_test, LinReg.predict(x_test))
        mse = mean_squared_error(y_test, LinReg.predict(x_test))
        rmse = np.sqrt(mse)
        coefficient = LinReg.coef_[0]
        intercept = LinReg.intercept_
        rss = np.sum((y_test - LinReg.predict(x_test)) ** 2)
        rmspe = np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        model = LinReg

        y_pred = LinReg.predict(x)

        return r2, mse, rmse, coefficient, intercept, rss, rmspe, mape, model, y_pred

    def _crossvalidation_method(self, x, y):
        """ Cross-validation method """
        if self._set_seed is None:
            if self._df.shape[0] / 2 >= self._k:
                kf = KFold(n_splits=self._k, shuffle=True)
            else:
                raise ValueError(f"Argument 'k' must be in the range [2, {int(self._df.shape[0] / 2)}]")
        else:
            if 0 <= self._set_seed <= 4294967295:
                if self._df.shape[0] / 2 >= self._k:
                    kf = KFold(n_splits=self._k, shuffle=True, random_state=self._set_seed)
                else:
                    raise ValueError(f"Argument 'k' must be in the range [2, {int(self._df.shape[0] / 2)}]")

        LinReg = LinearRegression()

        r2s = []
        mses = []
        rmses = []
        coefficients = []
        intercepts = []
        rsss = []
        rmspes = []
        mapes = []

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
            rsss.append(np.sum((y_test - y_pred) ** 2))
            rmspes.append(np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100)
            mapes.append(mean_absolute_percentage_error(y_test, y_pred) * 100)

        r2_mean = np.mean(r2s)
        mse_mean = np.mean(mses)
        rmse_mean = np.mean(rmses)
        coefficient_mean = np.mean(coefficients)
        intercept_mean = np.mean(intercepts)
        rss_mean = np.mean(rsss)
        rmspe_mean = np.mean(rmspes)
        mape_mean = np.mean(mapes)

        LinReg.fit(x, y)
        y_pred = LinReg.predict(x)
        model = LinReg

        return r2_mean, mse_mean, rmse_mean, coefficient_mean, intercept_mean, rss_mean, rmspe_mean, mape_mean, model, y_pred

    def plot_model(self):
        """ Plots the best feature against the response variable with the linear regression model """
        plt.scatter(self._df[self.best_feature], self._df[self._response], label="Data")
        x_range = np.linspace(self._df[self.best_feature].min(), self._df[self.best_feature].max(), 100).reshape(-1, 1)
        y_range_pred = self.coefficient * x_range + self.intercept
        plt.plot(x_range, y_range_pred, color="red", label="Regression Line")
        plt.xlabel(f"Feature: {self.best_feature}")
        plt.ylabel(f"Target: {self._response}")
        plt.title(f"{self.formula}")
        plt.legend()
        plt.show()


class BestMultipleLinearRegression:
    def __init__(self, df: pd.DataFrame, response: str, set_seed: int = None, divide_method: str = "train_test",
                 k: int = 5, test_size: float = 0.2):
        """
        The objects are models of multiple linear regression for the dependent variable and the independent variables
        that best fits it, using the r-squared coefficient.
        :param df: pandas DataFrame -> Input data frame
        :param response: string -> Name of dependent variable
        :param set_seed: int -> Seed number for reproducibility
        :param divide_method: 'train_test' -> Split data with train and test method
                              'crossvalidation' -> Split data with crossvalidation method
        :param k: int -> Folds in crossvalidation
        :param test_size: float -> Size of the test set
        """
        self._df = df
        self._response = response
        self._set_seed = set_seed
        self._divide_method = divide_method
        self._k = k
        self._test_size = test_size

        self._validate_inputs()
        self._find_best_feature()

    def _validate_inputs(self):
        dic = {
            "df": ["DataFrame", self._df],
            "response": ["str", self._response],
            "set_seed": ["NoneType or int", self._set_seed],
            "divide_method": ["NoneType or str", self._divide_method],
            "k": ["int", self._k],
            "test_size": ["float", self._test_size]
        }
        _validate_class_argument_types(dic)

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

        features_combs = _generate_independent_variables_combinations_list(features)

        for features_comb in features_combs:
            x = self._df[list(features_comb)].values

            r2, mse, rmse, coefficients, intercept, rss, rmspe, mape, model, y_pred = self._choose_divide_method(x, y)

            if r2 > self._best_feature_dic["r-squared"]:
                self._best_feature_dic["r-squared"] = r2
                self._best_feature_dic["features"] = features_comb
                self._best_feature_dic["mse"] = mse
                self._best_feature_dic["rmse"] = rmse
                self._best_feature_dic["coefficients"] = coefficients
                self._best_feature_dic["intercept"] = intercept
                self._best_feature_dic["formula"] = _count_line_formula_2(coefficients, intercept, self._response,
                                                                          features_comb)
                self._best_feature_dic["rss"] = rss
                self._best_feature_dic["rmspe"] = rmspe
                self._best_feature_dic["mape"] = mape
                self._best_feature_dic["model"] = model
                self._best_feature_dic["y_pred"] = y_pred

        self.best_features = self._best_feature_dic["features"]
        self.r_squared = self._best_feature_dic["r-squared"]
        self.mse = self._best_feature_dic["mse"]
        self.rmse = self._best_feature_dic["rmse"]
        self.coefficients = self._best_feature_dic["coefficients"]
        self.intercept = self._best_feature_dic["intercept"]
        self.formula = self._best_feature_dic["formula"]
        self.rss = self._best_feature_dic["rss"]
        self.rmspe = self._best_feature_dic["rmspe"]
        self.mape = self._best_feature_dic["mape"]
        self.model = self._best_feature_dic["model"]
        self.y_pred = self._best_feature_dic["y_pred"]

    def _choose_divide_method(self, x, y):
        if self._divide_method == 'train_test':
            return self._train_test_method(x, y)
        elif self._divide_method == 'crossvalidation':
            return self._crossvalidation_method(x, y)
        else:
            raise ValueError(f"Argument 'divide_method' must be 'train_test' or 'crossvalidation'")

    def _train_test_method(self, x, y):
        """ Train-test split method """
        if self._set_seed is None:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self._test_size)
        else:
            if 0 <= self._set_seed <= 4294967295:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self._test_size,
                                                                    random_state=self._set_seed)
            else:
                raise ValueError(f"Argument 'set_seed' must be in the range [0, 4294967295]")

        LinReg = LinearRegression()
        LinReg.fit(x_train, y_train)
        y_pred = LinReg.predict(x_test)

        r2 = r2_score(y_test, LinReg.predict(x_test))
        mse = mean_squared_error(y_test, LinReg.predict(x_test))
        rmse = np.sqrt(mse)
        coefficients = LinReg.coef_
        intercept = LinReg.intercept_
        rss = np.sum((y_test - LinReg.predict(x_test)) ** 2)
        rmspe = np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        model = LinReg

        y_pred = LinReg.predict(x)

        return r2, mse, rmse, coefficients, intercept, rss, rmspe, mape, model, y_pred

    def _crossvalidation_method(self, x, y):
        """ Cross-validation method """
        if self._set_seed is None:
            if self._df.shape[0] / 2 >= self._k:
                kf = KFold(n_splits=self._k, shuffle=True)
            else:
                raise ValueError(f"Argument 'k' must be in the range [2, {int(self._df.shape[0] / 2)}]")
        else:
            if 0 <= self._set_seed <= 4294967295:
                if self._df.shape[0] / 2 >= self._k:
                    kf = KFold(n_splits=self._k, shuffle=True, random_state=self._set_seed)
                else:
                    raise ValueError(f"Argument 'k' must be in the range [2, {int(self._df.shape[0] / 2)}]")

        LinReg = LinearRegression()

        r2s = []
        mses = []
        rmses = []
        coefficients = []
        intercepts = []
        rsss = []
        rmspes = []
        mapes = []

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
            rsss.append(np.sum((y_test - y_pred) ** 2))
            rmspes.append(np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100)
            mapes.append(mean_absolute_percentage_error(y_test, y_pred) * 100)

        r2_mean = np.mean(r2s)
        mse_mean = np.mean(mses)
        rmse_mean = np.mean(rmses)
        coefficients_mean = np.mean(coefficients, axis=0)
        intercept_mean = np.mean(intercepts)
        rss_mean = np.mean(rsss)
        rmspe_mean = np.mean(rmspes)
        mape_mean = np.mean(mapes)

        LinReg.fit(x, y)
        y_pred = LinReg.predict(x)
        model = LinReg

        return r2_mean, mse_mean, rmse_mean, coefficients_mean, intercept_mean, rss_mean, rmspe_mean, mape_mean, model, y_pred

    def plot_model(self):
        """ Plots the actual vs predicted values """
        y = self._df[self._response].values
        plt.scatter(self.y_pred, y, label='Data')
        plt.plot([y.min(), y.max()] , [y.min(), y.max()], color='red', label='Model')
        plt.title(self.formula)
        plt.xlabel('Predicted data')
        plt.ylabel('Real data')
        plt.legend()
        plt.show()
