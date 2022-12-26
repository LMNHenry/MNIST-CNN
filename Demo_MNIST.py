import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_val,y_val=x_train[50000:60000,:],y_train[50000:60000]
x_train,y_train=x_train[:50000,:],y_train[:50000]
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_val=x_val.reshape(x_val.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
y_train=np_utils.to_categorical(y_train,10)
y_val=np_utils.to_categorical(y_val,10)
y_test=np_utils.to_categorical(y_test,10)
model=Sequential()
model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28,28,1)))
model.add(Conv2D(32, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
H = model.fit(x_train, y_train, validation_data=(x_val, y_val),batch_size=32, epochs=10, verbose=1)
fig = plt.figure()
numOfEpoch = 10
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
score = model.evaluate(x_test,y_test,verbose=0)
print(score)
plt.imshow(x_test[0].reshape(28,28), cmap='gray')
y_predict = model.predict(x_test[0].reshape(1,28,28,1))
print('Giá trị dự đoán: ', np.argmax(y_predict))
import pickle
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))