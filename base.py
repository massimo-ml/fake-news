from abc import ABC, abstractmethod

import numpy as np


class AbstractNewsClassifier(ABC):
    def __init__(self, *, metrics: list[str]):
        self._metrics: list[str] = metrics

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass


class AbstractNewsGenerator(ABC):
    @abstractmethod
    def generate(self, title: str) -> str:
        pass
