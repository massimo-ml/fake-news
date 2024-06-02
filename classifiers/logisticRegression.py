import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore
import joblib  # type: ignore

import sys

sys.path.append("..")

from fake_news.base import AbstractNewsClassifier


class LogisticRegressionNewsClassifier(AbstractNewsClassifier):
    def __init__(self, *, metrics: list[str]):
        super().__init__(metrics=metrics)
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def save_model(self, file_path: str):
        joblib.dump(self.model, file_path)
        print("Model saved successfully at:", file_path)

    def load_model(self, file_path: str):
        self.model = joblib.load(file_path)
        print("Model loaded successfully from:", file_path)
