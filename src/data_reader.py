import os
from PIL import Image
import numpy as np
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
import h5py

path = Path('../sample/ucf101')
data = []
label = []
cnt = 0
min_frame = 32

for entry in os.scandir(path):
    if (entry.name != '.DS_Store'):
        entry_name = path/entry.name
        for input in os.scandir(entry_name):
            impath = entry_name/input.name/'n_frames'
            # impath = path + "/" + entry.name + "/" + input.name + "/" + 'n_frames'
            val = int(open(impath, 'r').read())
            min_frame = min(min_frame, val)

for entry in os.scandir(path):
    if (entry.name != '.DS_Store'):
        entry_name = path/entry.name
        print (entry_name.resolve())
        for input in os.scandir(entry_name):
            data_inner = []
            # print ("input", input)
            total_frame = int(open(entry_name/input.name/'n_frames','r').read())
            # print (total_frame)
            # l = (total_frame - min_frame)//2
            sel = total_frame//min_frame
            i = total_frame % min_frame
            for count in range(min_frame):
                if os.name == 'nt':
                    im_path = glob(str((entry_name/input.name).resolve())+"\\"+'image_*'+str(i)+'.jpg')[0]
                else:
                    im_path = glob(str((entry_name/input.name).resolve()) + "/" + 'image_*' + str(i) + '.jpg')[0]
                img = Image.open(im_path)
                img.load()
                img_np = np.asarray(img, dtype="int32")
                # print (img_np.shape)
                data_inner.append(img_np)
                i += sel
            data_inner = np.array(data_inner)
            # print (data_inner.shape)
            data.append(data_inner)
            label.append(cnt)
    cnt += 1
data = np.asarray(data)
label = np.asarray(label)
print("Data: ",data.shape)
print("Label: ",label.shape)

x_train, x_test, y_train, y_test = train_test_split(data,label,shuffle=True)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

h5f = h5py.File('../sample/data.h5', 'w')
h5f.create_dataset('x_train', data=x_train)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('x_test', data=x_test)
h5f.create_dataset('y_test', data=y_test)
h5f.close()