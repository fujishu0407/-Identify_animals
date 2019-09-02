from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.datasets import cifar10
from keras.utils import np_utils

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

print(y_train)
for i in range(len(y_train)):
    y_train[i][0] = 0
    y_train[i][1] = 0
    y_train[i][8] = 0
    y_train[i][9] = 0

for i in range(len(y_test)):
    y_test[i][0] = 0
    y_test[i][1] = 0
    y_test[i][8] = 0
    y_test[i][9] = 0

print(y_train)
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
open('cifar10_cnn.json',"w").write(json_string)
model.save_weights('cifar10_cnn.h5')

#モデルの表示
model.summary()

#評価
score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

#dogとcat以外を0にして学習する?
#取得するデータを選びたい categoricalで挑戦

#Test loss: 3.8658203319549562
#Test accuracy: 0.3112