import itertools
import pandas as pd  # type: ignore
import numpy as np

from typing import Any

import fake_news.base as base

from fake_news.data_preprocessing import DataPreprocessingBeforeClassifiers


def _preprocessArticle(
    article: str, preprocessor: DataPreprocessingBeforeClassifiers, type: str
) -> np.ndarray:
    """
    Embeds the generated article into a numpy array to make it compatible with classifiers
    """
    if type == "ml":
        return preprocessor.transform_ml(article)
    elif type == "dl":
        return preprocessor.transform_dl(article)
    else:
        raise ValueError(
            f"Invalid transform 'type'={type}. Must be one of ['ml', 'dl']"
        )


def evaluateGenerator(
    generator: base.AbstractNewsGenerator,
    classifiers: dict[str, tuple[base.AbstractNewsClassifier, str]],
    testTitles: list[str],
    paramValues: dict[str, Any],
    dataPreprocessor: DataPreprocessingBeforeClassifiers,
) -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the accuracies of the classifiers
    on the data generated by the generator

    Parameters
    ========
    generator: base.AbstractNewsGenerator - Fake news generator
    classifiers: dict[str, Tuple[base.AbstractNewsClassifier, str]] - A dict of tuples to evaluate the generator with, the key is the name of the classifier.
    The tuple contains two values - the first one is a classifier, the second one is the type of classifier ('dl' for neural network classifiers, 'ml' for everything else)
    testTitles: list[str] - A list of titles to pass to the generator
    paramValues: dict[str, Any] - A dictionary with possible generation parameters for the generator model
    dataPreprocessor: DataPreprocessingBeforeClassifiers - A DataPreprocessingBeforeClassifiers object with trained/loaded tokenizers

    Output
    ========
    A df.DataFrame with the "accuracy" of generation (that is, the percentage of generated articles classified as fake news by each classfier)

    """
    res: dict[str, list] = {
        colName: [] for colName in [p for p in paramValues] + [c for c in classifiers]
    }  # Params used as indices
    paramsCombinations = [
        {paramName: comb[i] for i, paramName in enumerate(paramValues)}
        for comb in list(itertools.product(*[paramValues[i] for i in paramValues]))
    ]  # Generate all possible generation parameters combinations

    for paramCombination in paramsCombinations:
        for paramName in paramCombination:
            generator.setGenerationParameter(
                paramName=paramName, value=paramCombination[paramName]
            )
            res[paramName].append(paramCombination[paramName])
        fakeNewsCountPerClassifier = {
            classifierName: 0 for classifierName in classifiers
        }
        for testTitle in testTitles:
            article = generator.generate(testTitle)
            for classifierName in classifiers:
                classification = classifiers[classifierName][0].predict(
                    _preprocessArticle(
                        article, dataPreprocessor, classifiers[classifierName][1]
                    )
                )
                if classification[0] == 1:
                    fakeNewsCountPerClassifier[classifierName] += 1
        for classifierName in classifiers:
            res[classifierName].append(
                fakeNewsCountPerClassifier[classifierName] / len(testTitles)
            )
    df = pd.DataFrame(data=res).set_index([p for p in paramValues])
    return df


# EXAMPLE OF USAGE

# mbCl = MultinomialNaiveBayesClassifier(metrics=[])
# mbCl.load_model("classifiers/naivebayes.pkl")
# rdFor = RandomForestClassifierClass(metrics=[])
# rdFor.load_model("classifiers/rf_model.pkl")
# cnnM = ConvolutionalNeuralNetworkClassifier(metrics=[])
# cnnM.load_model('classifiers/cnn.h5')
# dp = DataPreprocessingBeforeClassifiers(
#    "C:\\Users\\Kajetan\\AppData\\Roaming\\nltk_data"
# )
# dp.load_tokenizers(
#    "classifiers/tokenizers/ml_tokenizer.pickle",
#    "classifiers/tokenizers/dl_tokenizer.pickle",
# )
# df = evaluateGenerator(
#    DummyGenerator(),
#    {"mb_classifier": (mbCl, "ml"), "rf_classifier": (rdFor, "ml"), 'cnn': (cnnM, 'dl')},
#    ["title1", "title2", "title3"],
#    {"temp": [0, 1, 2, 3, 4]},
#    dp,
# )
