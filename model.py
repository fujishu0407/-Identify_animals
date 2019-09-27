from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten

class Model():
    def __init__(self):
        self.model=Sequential()
    
    def add(self,layer):
        self.model.add(layer)


    def cnn(self):
        self.model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32,(3,3),padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64,(3,3),padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64,(3,3),padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10,activation='softmax'))
        
        return self.model
        
#modelを呼び出した時に子要素としてmodelを持っているせいでfitなどが違和感