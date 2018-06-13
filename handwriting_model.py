import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# show sample images to make sure data test loads correctly:
#
# plt.subplot(221)
# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(X_train[5000], cmap=plt.get_cmap('gray'))
# # show the plot
# plt.show()


num_pixels = X_train.shape[1] * X_train.shape[2]
# num_pixels = 28 * 28 = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_train = X_train / 255
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def create_model():
    model = Sequential()

    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Scores: ", (scores))

model.save('handwriting_model.h5')
