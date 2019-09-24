from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np

(x_train,y_train),(x_test,y0_test)=cifar10.load_data()

x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y0_test,10)


for i in reversed(range(len(y_train))):
    if y_train[i][0] == 1 or y_train[i][1] == 1 or y_train[i][8] == 1 or y_train[i][9] == 1:
        y_train = np.delete(y_train, i, 0)
        x_train = np.delete(x_train, i, 0)
        print("finished",  i, " / 50000")


for i in reversed(range(len(y_test))):
    if y_test[i][0] == 1 or y_test[i][1] == 1 or y_test[i][8] == 1 or y_test[i][9] == 1:
        y_test = np.delete(y_test, i, 0)
        x_test = np.delete(x_test, i, 0)
        y0_test = np.delete(y0_test, i, 0)
        print("finished",  i, " / 10000")

data = [x_train, y_train, x_test, y_test]

np.save("animal_data.npy", data )
