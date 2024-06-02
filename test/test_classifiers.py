import unittest
import numpy as np

import fake_news.base as base
from fake_news.classifiers import (
    NaiveBayes,
    CNN,
    logisticRegression,
    LSTM,
    RandomForest,
    RNN,
    SVM,
)


class TestClassifier(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.mockInputMl = np.zeros((1, 10000))
        self.mockInputDl = np.zeros((1, 500))

    def _singleClassifierTest(
        self, classifier: base.AbstractNewsClassifier, input: np.ndarray
    ) -> None:
        pred = classifier.predict(input)
        self.assertEqual(pred.shape, (1,))

    def testNaiveBayesOutputShape(self):
        model = NaiveBayes.MultinomialNaiveBayesClassifier(metrics=[])
        model.load_model("fake_news/classifiers/naivebayes.pkl")
        self._singleClassifierTest(model, self.mockInputMl)

    def testLogisticRegressionOutputShape(self):
        model = logisticRegression.LogisticRegressionNewsClassifier(metrics=[])
        model.load_model("fake_news/classifiers/logisticregression.pkl")
        self._singleClassifierTest(model, self.mockInputMl)

    def testRandomForestOutputShape(self):
        model = RandomForest.RandomForestClassifierClass(metrics=[])
        model.load_model("fake_news/classifiers/rf_model.pkl")
        self._singleClassifierTest(model, self.mockInputMl)

    def testSVMOutputShape(self):
        model = SVM.SupportVectorMachineClassifier(metrics=[])
        model.load_model("fake_news/classifiers/svm_model.pkl")
        self._singleClassifierTest(model, self.mockInputMl)

    def testCNNOutputShape(self):
        model = CNN.ConvolutionalNeuralNetworkClassifier(metrics=[])
        model.load_model("fake_news/classifiers/cnn.h5")
        self._singleClassifierTest(model, self.mockInputDl)

    def testLSTMOutputShape(self):
        model = LSTM.LSTMClassifier(metrics=[])
        model.load_model("fake_news/classifiers/lstm_model.h5")
        self._singleClassifierTest(model, self.mockInputDl)

    def testRNNOutputShape(self):
        model = RNN.RecurrentNeuralNetworkClassifier(metrics=[])
        model.load_model("fake_news/classifiers/rnn.h5")
        self._singleClassifierTest(model, self.mockInputDl)


if __name__ == "__main__":
    unittest.main()
