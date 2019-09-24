from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.datasets import cifar10
from keras.utils import np_utils
import os
import numpy as np


data = np.load("animal_data.npy")


x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]



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
