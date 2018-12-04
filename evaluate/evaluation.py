import h5py

from src.video_classifier import VideoClassifier

def evaluate(path, supervised):
    with h5py.File(path, 'r') as hf:
        x_train = hf['x_train'][:]
        y_train = hf['y_train'][:]
        x_test = hf['x_test'][:]
        y_test = hf['y_test'][:]
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    supervised.fit(x_train, y_train, time_limit=60*60*6)
    supervised.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    return supervised.evaluate(x_test, y_test)


if __name__ == '__main__':
    print("Accuracy: ", evaluate('../sample/data.h5', VideoClassifier()))