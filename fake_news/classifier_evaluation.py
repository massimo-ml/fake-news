import pandas as pd  # type: ignore
import numpy as np
import scipy.sparse  # type: ignore

import fake_news.base as base
from fake_news.data_preprocessing import DataPreprocessingBeforeClassifiers
from fake_news.metrics import calculate_metrics, METRIC

import logging


def _fit_classifier(
    classifier: base.AbstractNewsClassifier,
    clf_type: str,
    preprocessor: DataPreprocessingBeforeClassifiers,
    input_df: pd.DataFrame,
):
    if clf_type == "ml":
        preprocess_func = preprocessor.transform_ml
        stack_func = scipy.sparse.vstack
    elif clf_type == "dl":
        preprocess_func = preprocessor.transform_dl
        stack_func = np.array

    x: scipy.sparse.spmatrix | np.ndarray
    x = stack_func([preprocess_func(article) for article in input_df["text"]])
    y = input_df["label"].values
    classifier.fit(x, y)


def _predict_classifier(
    classifier: base.AbstractNewsClassifier,
    clf_type: str,
    preprocessor: DataPreprocessingBeforeClassifiers,
    input_df: pd.DataFrame,
):
    if clf_type == "ml":
        preprocess_func = preprocessor.transform_ml
        stack_func = scipy.sparse.vstack
    elif clf_type == "dl":
        preprocess_func = preprocessor.transform_dl
        stack_func = np.array

    x: np.ndarray = stack_func(
        [preprocess_func(article) for article in input_df["text"]]
    )

    return classifier.predict(x)


def evaluate_classifiers(
    classifiers: list[tuple[type[base.AbstractNewsClassifier], str]],
    train_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: list[METRIC],
    orig_tokenizer_paths: tuple[str, str],
    combined_tokenizer_paths: tuple[str, str],
    nltk_data_path: str | None = None,
):
    orig_preprocessor = DataPreprocessingBeforeClassifiers(nltk_data_path)
    combined_preprocessor = DataPreprocessingBeforeClassifiers(nltk_data_path)

    logging.info("Loading tokenizers")
    orig_preprocessor.load_tokenizers(*orig_tokenizer_paths)
    combined_preprocessor.load_tokenizers(*combined_tokenizer_paths)
    # orig_preprocessor.fit_tokenizers_df(train_df, test_df)
    # combined_preprocessor.fit_tokenizers_df(train_df, test_df)

    metrics_results = []

    for classifier_cls, clf_type in classifiers:
        logging.info(f"Started evaluating {classifier_cls}")

        # Train classifier without synthetic
        logging.info("Fitting and predicting on original data")
        orig_classifier = classifier_cls()
        _fit_classifier(orig_classifier, clf_type, orig_preprocessor, train_df)
        # Predict
        orig_pred = _predict_classifier(
            orig_classifier, clf_type, orig_preprocessor, test_df
        )
        del orig_classifier

        # Train classifier with synthetic
        logging.info("Fitting and predicting on combined data")
        combined_classifier = classifier_cls()
        _fit_classifier(
            combined_classifier,
            clf_type,
            combined_preprocessor,
            pd.concat([train_df, synth_df], axis=0).reset_index(drop=True),
        )
        # Predict
        combined_pred = _predict_classifier(
            combined_classifier, clf_type, orig_preprocessor, test_df
        )
        del combined_classifier

        # Evaluate on test set
        logging.info("Calculating metrics")

        orig_metrics = calculate_metrics(
            test_df["label"].values, orig_pred, metrics=metrics
        )
        combined_metrics = calculate_metrics(
            test_df["label"].values, combined_pred, metrics=metrics
        )
        metrics_results.append((orig_metrics, combined_metrics))

    return metrics_results


if __name__ == "__main__":
    from fake_news.classifiers.NaiveBayes import (
        MultinomialNaiveBayesClassifier,
    )

    logging.getLogger().setLevel(logging.INFO)

    train_df = pd.read_csv(
        R"D:\Documents\University\Semester6\NLP\fake-news\data\WELFake_clean_train.csv"  # noqa
    )
    test_df = pd.read_csv(
        R"D:\Documents\University\Semester6\NLP\fake-news\data\WELFake_clean_test.csv"  # noqa
    )

    tokenizer_paths = (
        R"D:\Documents\University\Semester6\NLP\fake-news\fake_news\classifiers\tokenizers\ml_tokenizer.pickle",  # noqa
        R"D:\Documents\University\Semester6\NLP\fake-news\fake_news\classifiers\tokenizers\dl_tokenizer.pickle",  # noqa
    )

    results = evaluate_classifiers(
        classifiers=[(MultinomialNaiveBayesClassifier, "ml")],
        train_df=test_df,  # train_df,
        synth_df=test_df,
        test_df=test_df,
        orig_tokenizer_paths=tokenizer_paths,
        combined_tokenizer_paths=tokenizer_paths,
        metrics=["acc", "auc", "f1"],
    )
    print(results)
