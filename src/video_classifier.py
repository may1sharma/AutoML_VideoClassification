import os
import torch
import numpy as np
from skimage.transform import resize
from functools import reduce

from simple_nn import Net

from autokeras.supervised import Supervised
from autokeras.preprocessor import OneHotEncoder, ImageDataTransformer
from autokeras.nn.model_trainer import ModelTrainer
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.utils import pickle_to_file, pickle_from_file


class VideoClassifier(Supervised):
    def __init__(self):
        super().__init__(verbose=False)
        self.labels = None
        self.net = Net()
        self.augment = None
        self.Epochs = 2
        self.encoder = OneHotEncoder()
        self.path = '../temp'

    def evaluate(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_test, y_predict)

    def predict(self, x_test):
        classifier = pickle_from_file(os.path.join(self.path, 'classifier'))
        self.__dict__ = classifier.__dict__
        self.net.eval()
        test_data = self.preprocess(x_test)
        test_data = self.data_transformer.transform_test(test_data)
        outputs = []
        with torch.no_grad():
            for index, inputs in enumerate(test_data):
                outputs.append(self.net(inputs).numpy())
        output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
        predicted = self.encoder.inverse_transform(output)
        return predicted

    def fit(self, x_train=None, y_train=None, time_limit=None):
        x_train = self.preprocess(x_train)

        x_valid, y_valid = [], []
        x_valid.append(x_train[0])
        x_valid.append(x_train[15])
        y_valid.append(y_train[0])
        y_valid.append(y_train[15])

        x_train, y_train = np.array(x_train[1:15]), np.array(y_train[1:15])
        x_valid, y_valid = np.array(x_valid), np.array(y_valid)

        self.encoder.fit(y_train)
        y_train = self.encoder.transform(y_train)
        y_valid = self.encoder.transform(y_valid)

        self.data_transformer = ImageDataTransformer(x_train, augment=self.augment)
        train_data = self.data_transformer.transform_train(x_train, y_train, batch_size=1)
        test_data = self.data_transformer.transform_test(x_valid, y_valid, batch_size=1)

        self.model_trainer = ModelTrainer(self.net,
                                     path=self.path,
                                     loss_function=classification_loss,
                                     metric=Accuracy,
                                     train_data=train_data,
                                     test_data=test_data,
                                     verbose=True)

        # Searching for best epoch as a hyperparameter
        accuracy = 0
        y_valid = self.encoder.inverse_transform(y_valid)
        for epoch in [2, 4, 6]:
            self.model_trainer.train_model(epoch, 2)
            outputs = []
            with torch.no_grad():
                for index, (inputs, _) in enumerate(test_data):
                    outputs.append(self.net(inputs).numpy())
            output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
            pred_valid = self.encoder.inverse_transform(output)
            accu = self.metric().evaluate(y_valid, pred_valid)

            if accu > accuracy:
                self.Epochs = epoch

        print('Finished Fit')

    def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=True):
        x_train = self.preprocess(x_train)
        x_test = self.preprocess(x_test)

        self.encoder.fit(y_train)
        y_train = self.encoder.transform(y_train)
        y_test = self.encoder.transform(y_test)

        self.data_transformer = ImageDataTransformer(x_train, augment=self.augment)
        train_data = self.data_transformer.transform_train(x_train, y_train, batch_size=1)
        test_data = self.data_transformer.transform_test(x_test, y_test, batch_size=1)

        pickle_to_file(self, os.path.join(self.path, 'classifier'))

        self.model_trainer = ModelTrainer(self.net,
                                          path=self.path,
                                          loss_function=classification_loss,
                                          metric=Accuracy,
                                          train_data=train_data,
                                          test_data=test_data,
                                          verbose=True)

        self.model_trainer.train_model(self.Epochs, 2)
        print('Finished Final Fit')

    def preprocess(self, data):
        result = []
        for video in data:
            clip = np.array([resize(frame, output_shape=(112, 200), preserve_range=True) for frame in video])
            clip = clip[:, :, 44:44 + 112, :]  # crop centrally
            result.append(clip)
        result = np.array(result)
        return result

    @property
    def metric(self):
        return Accuracy