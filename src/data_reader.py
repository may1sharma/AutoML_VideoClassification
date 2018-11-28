import os
from PIL import Image
import numpy as np
import pickle
from glob import glob
from sklearn.model_selection import train_test_split

path = '../sample/ucf101'
data = []
label = []
cnt=0
min_frame = 50

for entry in os.scandir(path):
    if (entry.name != '.DS_Store'):
        for input in os.scandir(path + "/" + entry.name):
            impath = path + "/" + entry.name + "/" + input.name + "/" + 'n_frames'
            val = int(open(impath, 'r').read())
            min_frame = min(min_frame, val)

for entry in os.scandir(path):
    if (entry.name != '.DS_Store'):
        for input in os.scandir(path + "/" + entry.name):
            data_inner = []
            total_current_frame = int(open(path + "/" + entry.name + "/" + input.name + '/n_frames','r').read())
            l = (total_current_frame - min_frame)//2
            for i in range(l,l+min_frame):
                impath = glob(path + "/" + entry.name + "/" + input.name + "/" + 'image_*'+str(i)+'.jpg')[0]
                img = Image.open(impath)
                img.load()
                img_np = np.asarray(img, dtype="int32")
                data_inner.append(img_np)
            data_inner = np.asarray(data_inner)
            data.append(data_inner)
            label.append(cnt)
    cnt += 1
data = np.asarray(data)
label = np.asarray(label)
print("Data: ",data.shape)
print("Label: ",label.shape)

x_train, x_test, y_train, y_test = train_test_split(data,label,shuffle=True)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


save = (x_train, y_train), (x_test, y_test)

with open('../sample/data_new.pkl','wb') as f:
    pickle.dump(save, f)