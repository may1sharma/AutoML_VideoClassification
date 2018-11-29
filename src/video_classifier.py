import os
import torch
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from functools import reduce
import heapq
import time

from autokeras.supervised import Supervised
from autokeras.preprocessor import OneHotEncoder, ImageDataTransformer
from autokeras.nn.model_trainer import ModelTrainer
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.utils import pickle_to_file, pickle_from_file
from autokeras.nn.generator import CnnGenerator


class VideoClassifier(Supervised):
    def __init__(self):
        super().__init__(verbose=False)
        self.labels = None
        self.net = None
        self.augment = None
        self.Length = 3
        self.Width = 4
        self.Epochs = 10
        self.encoder = OneHotEncoder()
        self.path = '../temp'
        self.capacity = 50

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

    def fit(self, x_train=None, y_train=None, time_limit=60*60*6):
        end_time = time.time()+time_limit

        x_train = self.preprocess(x_train)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, shuffle=True)
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_valid, y_valid = np.array(x_valid), np.array(y_valid)

        self.encoder.fit(y_train)
        y_train = self.encoder.transform(y_train)
        y_valid = self.encoder.transform(y_valid)

        self.data_transformer = ImageDataTransformer(x_train, augment=self.augment)
        train_data = self.data_transformer.transform_train(x_train, y_train, batch_size=1)
        test_data = self.data_transformer.transform_test(x_valid, y_valid, batch_size=1)

        y_valid = self.encoder.inverse_transform(y_valid)

        visited = set()
        pq = []
        trainingQ = [(self.Length, self.Width, self.Epochs)]
        accuracy = 0.0

        while trainingQ:
            inc = False
            for len, width, epoch in trainingQ:
                if time.time() < end_time and (len, width, epoch) not in visited:
                    visited.add((len, width, epoch))
                    try:
                        net = CnnGenerator(self.encoder.n_classes, x_train.shape[1:])\
                            .generate(model_len=len, model_width=width).produce_model()

                        model_trainer = ModelTrainer(net,
                                                 path=self.path,
                                                 loss_function=classification_loss,
                                                 metric=Accuracy,
                                                 train_data=train_data,
                                                 test_data=test_data,
                                                 verbose=True)
                        model_trainer.train_model(epoch, 3)

                        outputs = []
                        with torch.no_grad():
                            for index, (inputs, _) in enumerate(test_data):
                                outputs.append(net(inputs).numpy())
                        output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
                        pred_valid = self.encoder.inverse_transform(output)
                        accu = self.metric().evaluate(y_valid, pred_valid)

                        pq.append((-accu, (len, width, epoch)))
                        if len(pq) > self.capacity:
                            heapq.heapify(pq)
                            pq.remove(heapq.nlargest(1, pq))
                        if accu > accuracy:
                            self.Epochs = epoch
                            self.Length = len
                            self.Width = width
                            accuracy = accu
                            inc = True
                    except Exception as e:
                        print(e)

            if not inc:
                if not pq:
                    break
                heapq.heapify(pq)
                _, (nexlen, nexwidth, nexepoch) = heapq.heappop(pq)
            else:
                nexlen, nexwidth, nexepoch = self.Length, self.Width, self.Epochs

            # Create children
            trainingQ = []
            trainingQ.append((nexlen+1, nexwidth, nexepoch))
            trainingQ.append((nexlen, nexwidth*2, nexepoch))
            trainingQ.append((nexlen, nexwidth, nexepoch+5))
            trainingQ.append((nexlen+2, nexwidth, nexepoch+3))
            trainingQ.append((nexlen, nexwidth*2, nexepoch+2))
            trainingQ.append((nexlen+1, nexwidth, nexepoch+3))

        print('Finished Fit')
        print("Optimal Conv3D Network Parameters:")
        print("Number of Layers (Length):", self.Length)
        print("Number of Filters (Width):", self.Width)
        print("Number of Epochs", self.Epochs)
        print()


    def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=True):
        x_train = self.preprocess(x_train)
        x_test = self.preprocess(x_test)

        self.encoder.fit(y_train)
        y_train = self.encoder.transform(y_train)
        y_test = self.encoder.transform(y_test)

        self.data_transformer = ImageDataTransformer(x_train, augment=self.augment)
        train_data = self.data_transformer.transform_train(x_train, y_train, batch_size=1)
        test_data = self.data_transformer.transform_test(x_test, y_test, batch_size=1)

        self.net = CnnGenerator(self.encoder.n_classes, x_train.shape[1:]) \
            .generate(model_len=self.Length, model_width=self.Width).produce_model()

        pickle_to_file(self, os.path.join(self.path, 'classifier'))

        self.model_trainer = ModelTrainer(self.net,
                                          path=self.path,
                                          loss_function=classification_loss,
                                          metric=Accuracy,
                                          train_data=train_data,
                                          test_data=test_data,
                                          verbose=True)

        self.model_trainer.train_model(self.Epochs, 3)
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