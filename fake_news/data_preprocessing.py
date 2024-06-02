import pandas as pd  # type: ignore
import numpy as np
import re
import pickle
import nltk  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import (  # type: ignore
    pad_sequences,
)


class DataPreprocessingBeforeClassifiers:
    def __init__(self, nltk_data_path: str) -> None:
        self.dl_seq_maxlen = 500
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.ml_vectorizer: TfidfVectorizer | None = None
        self.dl_vectorizer: Tokenizer | None = None
        nltk.data.path.append(nltk_data_path)

    def fit_tokenizers(self, df_train_path: str, df_test_path: str):
        self.df_train = pd.read_csv(df_train_path)
        self.df_test = pd.read_csv(df_test_path)

        self.df_train["title"] = self.df_train["title"].apply(self.clean_text)
        self.df_train["text"] = self.df_train["text"].apply(self.clean_text)
        self.df_test["title"] = self.df_test["title"].apply(self.clean_text)
        self.df_test["text"] = self.df_test["text"].apply(self.clean_text)

        self.df_train["title"] = self.df_train["title"].apply(word_tokenize)
        self.df_train["text"] = self.df_train["text"].apply(word_tokenize)
        self.df_test["title"] = self.df_test["title"].apply(word_tokenize)
        self.df_test["text"] = self.df_test["text"].apply(word_tokenize)

        self.df_train["title"] = self.df_train["title"].apply(
            lambda x: [word for word in x if word not in self.stop_words]
        )
        self.df_train["text"] = self.df_train["text"].apply(
            lambda x: [word for word in x if word not in self.stop_words]
        )
        self.df_test["title"] = self.df_test["title"].apply(
            lambda x: [word for word in x if word not in self.stop_words]
        )
        self.df_test["text"] = self.df_test["text"].apply(
            lambda x: [word for word in x if word not in self.stop_words]
        )

        self.df_train["title"] = self.df_train["title"].apply(
            lambda x: [self.lemmatizer.lemmatize(token) for token in x]
        )
        self.df_train["text"] = self.df_train["text"].apply(
            lambda x: [self.lemmatizer.lemmatize(token) for token in x]
        )
        self.df_test["title"] = self.df_test["title"].apply(
            lambda x: [self.lemmatizer.lemmatize(token) for token in x]
        )
        self.df_test["text"] = self.df_test["text"].apply(
            lambda x: [self.lemmatizer.lemmatize(token) for token in x]
        )

        self.df_train["joined"] = (
            self.df_train["title"].apply(lambda x: " ".join(x))
            + " "
            + self.df_train["text"].apply(lambda x: " ".join(x))
        )
        self.df_test["joined"] = (
            self.df_test["title"].apply(lambda x: " ".join(x))
            + " "
            + self.df_test["text"].apply(lambda x: " ".join(x))
        )

        self.X_train = self.df_train["joined"]
        self.y_train = self.df_train["label"]
        self.X_test = self.df_test["joined"]
        self.y_test = self.df_test["label"]

        self.ml_vectorizer = TfidfVectorizer(max_features=10000)
        self.ml_vectorizer.fit(self.X_train)

        self.dl_vectorizer = Tokenizer(num_words=10000)
        self.dl_vectorizer.fit_on_texts(self.X_train)

    def save_tokenizers(
        self, ml_vectorizer_path: str, dl_vectorizer_path: str
    ):
        with open(ml_vectorizer_path, "wb") as f:
            pickle.dump(self.ml_vectorizer, f)
        with open(dl_vectorizer_path, "wb") as f:
            pickle.dump(self.dl_vectorizer, f)

    def load_tokenizers(
        self, ml_vectorizer_path: str, dl_vectorizer_path: str
    ):
        with open(ml_vectorizer_path, "rb") as f:
            self.ml_vectorizer = pickle.load(f)
        with open(dl_vectorizer_path, "rb") as f:
            self.dl_vectorizer = pickle.load(f)

    def clean_text(self, text: str) -> str:
        text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
        text = re.sub(
            r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE
        )  # Remove URLs
        text = re.sub(
            r"[^A-Za-z\s]", "", text
        )  # Remove special characters and numbers
        text = text.lower()  # Convert to lowercase
        return text

    def _tokenize_lemmatize_text(self, text: str) -> list:
        tokenized = word_tokenize(self.clean_text(text))
        tokenized = [t for t in tokenized if t not in self.stop_words]
        tokenized = [self.lemmatizer.lemmatize(t) for t in tokenized]
        return [" ".join(tokenized)]

    def transform_ml(
        self, text: str
    ) -> np.ndarray:  # Use this for inference of single examples
        assert (
            self.ml_vectorizer is not None
        ), "fit_tokenizers() should be called first"
        tokenized = self._tokenize_lemmatize_text(text)
        transformed = self.ml_vectorizer.transform(tokenized)
        return transformed

    def transform_dl(
        self, text: str
    ) -> np.ndarray:  # Use this for inference of single examples
        assert (
            self.dl_vectorizer is not None
        ), "fit_tokenizers() should be called first"
        tokenized = self._tokenize_lemmatize_text(text)
        transformed = self.dl_vectorizer.texts_to_sequences(tokenized)
        padded = pad_sequences(transformed, maxlen=self.dl_seq_maxlen)
        return padded

    def out_ml(self):
        X_train_tfidf = self.ml_vectorizer.transform(self.X_train)
        X_test_tfidf = self.ml_vectorizer.transform(self.X_test)
        return X_train_tfidf, self.y_train, X_test_tfidf, self.y_test

    def out_dl(self):
        X_train_seq = self.dl_vectorizer.texts_to_sequences(self.X_train)
        X_test_seq = self.dl_vectorizer.texts_to_sequences(self.X_test)
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.dl_seq_maxlen)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.dl_seq_maxlen)
        return X_train_pad, self.y_train, X_test_pad, self.y_test
