import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle

from collections import Counter
import sys

from sklearn.metrics import classification_report

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 21)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 11)
        self.fc1 = nn.Linear(20 * 30 * 6, 900)
        self.fc2 = nn.Linear(900, 200)
        self.fc3 = nn.Linear(200, 50)
        self.fc4 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 30 * 6)
        x = F.relu(self.fc1(x))
        x = F.dropout(F.relu(self.fc2(x),0.2))
        x = F.dropout(F.relu(self.fc3(x)),0.4)
        x = F.dropout(self.fc4(x), 0.2)
        return F.log_softmax(x, dim=1)


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

classes = ('ball', 'gym')

def train(path):
    (x_train, y_train), (x_test, y_test) = pickle.load(open(path, 'rb'))

    for epoch in range(5):  # loop over the dataset multiple times
        # running_loss = 0.0
        for i, data in enumerate(x_train):
            # get the inputs
            labels = y_train[i]

            data = data.reshape(16,3,240,320)

            inputs = torch.tensor(data)

            inputs = inputs.float()

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(inputs)

            llist = [labels]*16

            labels = torch.tensor(llist,dtype=torch.int64)
            labels = labels.view(-1)

            loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            print("Loss: ", loss.item())

    torch.save(net.state_dict(), 'model')
    print('Finished Training')


def predict(path):
    (x_train, y_train), (x_test, y_test) = pickle.load(open(path, 'rb'))
    net.load_state_dict(torch.load('model'))
    pred = []
    for i, data in enumerate(x_test):
        data = data.reshape(16, 3, 240, 320)
        inputs = torch.tensor(data)
        inputs = inputs.float()
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = Counter(predicted.numpy()).most_common(1)[0][0]
        # print('Predicted: ', classes[c])
        pred.append(c)
    print (y_test)
    print (pred)
    print(classification_report(y_test,pred))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == "--train":
        train('../sample/video_classification_data')
    elif cmd == "--test":
        predict('../sample/video_classification_data')
    else:
        print ("Invalid option. Enter --train or --test")