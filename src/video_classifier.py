import numpy as np
from autokeras.metric import Accuracy
from autokeras.supervised import Supervised


class VideoClassifier(Supervised):
    def __init__(self):
        super().__init__(verbose=False)
        self.labels = None

    def evaluate(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_test, y_predict)

    def predict(self, x_test):
        return np.random.choice(self.labels, len(x_test))

    def fit(self, x_train=None, y_train=None, time_limit=None):
        self.labels = list(set(np.array(y_train).flatten()))

    @property
    def metric(self):
        return Accuracy

    def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=True):
        pass