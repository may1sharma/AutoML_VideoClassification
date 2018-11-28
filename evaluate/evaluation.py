import pickle

from src.video_classifier import VideoClassifier


def evaluate(path, supervised):
    (x_train, y_train), (x_test, y_test) = pickle.load(open(path, 'rb'))
    supervised.fit(x_train, y_train, time_limit = 15 * 60)
    supervised.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    return supervised.evaluate(x_test, y_test)


if __name__ == '__main__':
    print("Accuracy: ",evaluate('../sample/sample/video_classification_data', VideoClassifier()))