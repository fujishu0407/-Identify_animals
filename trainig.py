from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.datasets import cifar10
from keras.utils import np_utils
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import random


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
    # y_train[i][0] = 0
    # y_train[i][1] = 0
    # y_train[i][8] = 0
    # y_train[i][9] = 0


for i in reversed(range(len(y_test))):
    if y_test[i][0] == 1 or y_test[i][1] == 1 or y_test[i][8] == 1 or y_test[i][9] == 1:
        y_test = np.delete(y_test, i, 0)
        x_test = np.delete(x_test, i, 0)
        y0_test = np.delete(y0_test, i, 0)
        print("finished",  i, " / 10000")
    # y_test[i][0] = 0
    # y_test[i][1] = 0
    # y_test[i][8] = 0
    # y_test[i][9] = 0


cifar10_labels = np.array(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])

pos = 1
index = random.randint(0, x_test.shape[0]/2)
i = index
# plt figure set to 16inch x 16inch(1600pixel x 1600 pixel). 
plt.figure(figsize=(16,5))

# draw cifar10 images and label names
for img in x_test[index:index+30]:
    plt.subplot(3, 10, pos)
    plt.imshow(img)
    plt.axis('off')
    plt.title( cifar10_labels[y0_test[i][0]] )
    pos += 1
    i += 1

plt.show()

model=Sequential()

model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#nb_epoch=20
history=model.fit(x_train,y_train,batch_size=128,nb_epoch=20,verbose=1,validation_split=0.2)

#モデルと重みを保存
json_string=model.to_json()
open('cifar10_animal_cnn.json',"w").write(json_string)
model.save_weights('cifar10_animal_cnn.h5')

#モデルの表示
model.summary()

#評価
score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

#Test loss: 3.8658203319549562
#Test accuracy: 0.3112
# Test loss: 0.482819014942646
# Test accuracy: 0.4454
#まだまだ減少中
