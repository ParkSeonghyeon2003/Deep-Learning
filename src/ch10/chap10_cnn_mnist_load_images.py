from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.utils import to_categorical
import os

img_rows=28
img_cols=28

img_dir_train = "data/mnist/training"
img_dir_test = "data/mnist/testing"

flist_train = os.listdir(img_dir_train)
flist_test = os.listdir(img_dir_test)

from keras.preprocessing import image

X_train = np.zeros(shape=(len(flist_train), 28, 28, 3))
y_train = np.zeros(shape=(len(flist_train)))
X_test = np.zeros(shape=(len(flist_test), 28, 28, 3))
y_test = np.zeros(shape=(len(flist_test)))

for idx, fname in enumerate(flist_train):
    img_path = os.path.join(img_dir_train, fname)
    img = image.load_img(img_path, target_size=(28, 28))
    img_array_train = image.img_to_array(img)
    imt_array_train = np.expand_dims(img_array_train, axis=0)
    X_train[idx] = img_array_train
    y_train[idx] = flist_train[idx][:1]

for idx, fname in enumerate(flist_test):
    img_path = os.path.join(img_dir_test, fname)
    img = image.load_img(img_path, target_size=(28, 28))
    img_array_test = image.img_to_array(img)
    imt_array_test = np.expand_dims(img_array_test, axis=0)
    X_test[idx] = img_array_test
    y_test[idx] = flist_test[idx][:1]

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# data augmentation 
datagen = ImageDataGenerator(
    zoom_range=0.2, 
    shear_range=0.2,
    rotation_range=10,
    fill_mode='nearest',
    validation_split = 0.2
    )

datagen.fit(X_train)

train_gen = datagen.flow(X_train, y_train, batch_size=60)

# fix random seed for reproducibility 
seed = 100 
np.random.seed(seed)
num_classes = 10

# create CNN model
def cnn_model():
    # define model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), 
                            padding='valid',
                            strides=(1, 1),
                            input_shape=(img_rows, img_cols, 3), 
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(127, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model


# build the model
model = cnn_model()

# Fit the model
disp = model.fit(train_gen, 
          validation_data=(X_test, y_test), 
          epochs=10,   # 100
          batch_size=200, 
          verbose=1)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("loss: %.2f" % scores[0])
print("acc: %.2f" % scores[1])

# summarize history for accuracy
plt.plot(disp.history['accuracy'])
plt.plot(disp.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
