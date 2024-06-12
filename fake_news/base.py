from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from transformers import GenerationConfig  # type: ignore


class AbstractNewsClassifier(ABC):
    def __init__(self, *, metrics: list[str] | None = None):
        self._metrics: list[str] | None = (
            metrics  # TODO: This attribute is never used??
        )

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
        self,
        titles: list[str],
        generation_config: GenerationConfig | dict[str, Any] | None = None,
    ) -> list[str]:
        pass
