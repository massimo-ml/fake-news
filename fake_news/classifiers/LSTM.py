import numpy as np
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import Embedding, Dense, LSTM  # type: ignore

from fake_news.base import AbstractNewsClassifier


class LSTMClassifier(AbstractNewsClassifier):
    def __init__(self, *, metrics: list[str] | None = None):
        super().__init__(metrics=metrics)
        self.model = Sequential()
        self.model.add(
            Embedding(input_dim=10000, output_dim=128, input_length=500)
        )
        self.model.add(LSTM(128))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.fit(x, y, epochs=10, batch_size=64, validation_split=0.2)

    def predict(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(x)
        y_pred = (y_pred > 0.5).astype(int)
        return y_pred[0]

    def save_model(self, file_path: str):
        self.model.save(file_path)
        print("Model saved successfully at:", file_path)

    def load_model(self, file_path: str):
        self.model = load_model(file_path)
        print("Model loaded successfully from:", file_path)
