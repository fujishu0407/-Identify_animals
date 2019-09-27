import os
import numpy as np
#from model import Model
import matplotlib.pyplot as plt

data = np.load("animal_data.npy")

x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]

model = Model().cnn()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,batch_size=128,nb_epoch=30,verbose=1,validation_split=0.2)

history.history.keys()

json_string=model.to_json()
open('cifar10_animal_cnn_tmp.json',"w").write(json_string)
model.save_weights('cifar10_animal_cnn_tmp.h5')

#モデルの表示
model.summary()

#グラフ化
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#評価
score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

