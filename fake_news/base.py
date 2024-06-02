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

    @abstractmethod
    def save_model(self, file_path: str) -> None:
        pass
    
    @abstractmethod
    def load_model(self, file_path: str) -> None:
        pass


class AbstractNewsGenerator(ABC):
    def __init__(self, **genParams):
        self._generationParams = genParams

    @abstractmethod
    def generate(self, title: str) -> str:
        pass

    def setGenerationParameter(self, paramName: str, value):
        """
        Value can be of any type, so I'm not adding a hint 
        """
        if paramName in self._generationParams:
            self._generationParams[paramName] = value
    


