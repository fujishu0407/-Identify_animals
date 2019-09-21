from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

category=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

img=load_img("images/horse1.jpeg")
temp_img = img.resize((32,32))

temp_img_array = img_to_array(temp_img)
temp_img_array = temp_img_array.astype('float32')/255.0
temp_img_array = temp_img_array.reshape((1,32,32,3))

json_string = open('cifar10_animal_cnn.json').read()
model = model_from_json(json_string)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.load_weights('cifar10_animal_cnn.h5')

img_pred=model.predict_classes(temp_img_array)

img_pred = img_pred[0]

print(category[img_pred])
plt.imshow(img)
plt.title(category[img_pred])
plt.show()