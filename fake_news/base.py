from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from transformers import GenerationConfig  # type: ignore


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
    @abstractmethod
    def generate(
        self, title: str, generation_config: GenerationConfig | dict[str, Any]
    ) -> str:
        pass
