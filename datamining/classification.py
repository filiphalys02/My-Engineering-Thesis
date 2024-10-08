import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from datamining._errors import _validate_inputs


class BestClassification:
    def __init__(self, df: pd.DataFrame, response: str, set_seed: int = None, divide_method: str = "train_test",
                 k: int = 5, test_size: float = 0.2):
        """
        The objects are classification models for the dependent variable and the independent variables
        that best classify it, using the accuracy score.
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
        self._find_best_model()

    def _validate_inputs(self):
        dic = {
            "df": ["DataFrame", self._df],
            "response": ["str", self._response],
            "set_seed": ["NoneType or int", self._set_seed],
            "divide_data": ["NoneType or str", self._divide_method],
            "k": ["int", self._k],
            "test_size": ["float", self._test_size],
        }
        _validate_inputs(dic)

    def _find_best_model(self):
        """ Finds the best classification model"""

        try:
            self._df = self._df.select_dtypes(exclude=['datetime'])
            self._df = self._df.select_dtypes(exclude=['timedelta'])
            features = list(self._df.columns)
            features.remove(self._response)
            if not isinstance(self._df[self._response].dtype, pd.CategoricalDtype):
                raise TypeError(f"There is no '{self._response}' categorical column in your Data Frame")
        except ValueError:
            raise ValueError(f"There is no '{self._response}' column in your Data Frame")

        X = self._df[features]
        y = self._df[self._response]

        X = pd.get_dummies(X, drop_first=True)

        self._best_model_dic = {"accurancy": -1}

        models = {
            "Logistic Regression": LogisticRegression(solver='liblinear'),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine": SVC()
        }

        for model_name, model in models.items():
            accuracy, model, y_pred = self._choose_divide_method(model, X, y)

            if accuracy > self._best_model_dic["accurancy"]:
                self._best_model_dic["accuracy"] = accuracy
                self._best_model_dic["model_name"] = model_name
                self._best_model_dic["model"] = model
                self._best_model_dic["y_pred"] = y_pred

        self.accuracy = self._best_model_dic["accuracy"]
        self.model_name = self._best_model_dic["model_name"]
        self.model = self._best_model_dic["model"]
        self.y_pred = self._best_model_dic["y_pred"]

    def _choose_divide_method(self, model, X, y):
        if self._divide_method == 'train_test':
            return self._train_test_method(model, X, y)
        elif self._divide_method == 'crossvalidation':
            return self._crossvalidation_method(model, X, y)
        else:
            raise ValueError(f"Argument 'divide_method' must be 'train_test' or 'crossvalidation'")

    def _train_test_method(self, model, X, y):
        """ Train-test split method """
        if self._set_seed is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._test_size)
        else:
            if 0 <= self._set_seed <= 4294967295:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._test_size, random_state=self._set_seed)
            else:
                raise ValueError("Seed must be between 0 and 4294967295")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        y_pred = model.predict(X)
        return accuracy, model, y_pred
