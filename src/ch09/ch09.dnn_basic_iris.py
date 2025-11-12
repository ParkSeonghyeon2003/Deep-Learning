# keras DNN example
# dataset : iris

# load required modules
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load dataset
dataframe = pd.read_csv("/home/shpark/Projects/Python/DKU-deep-learning/src/dku_deep_learning/ch09/iris.csv")
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)

# Divide train, test
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.4, random_state=321)

################################################################
# define model (DNN structure)
epochs = 50
batch_size = 10

model = Sequential()
model.add(Input(shape=(4,)))       # input layer (4 features) 
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()  # show model structure

# Compile model

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# model fitting (learning)
disp = model.fit(X_train, Y_train, 
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,        # print fitting process  
                 validation_data=(X_test, Y_test))

############################################################## 
# Test model
pred = model.predict(X_test)
print(pred)
y_classes = [np.argmax(y, axis=None, out=None) for y in pred]
print(y_classes)   # result of prediction

# model performance
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# summarize history for accuracy
plt.plot(disp.history['accuracy'])
plt.plot(disp.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

############################################################## 
# model weights
for lay in model.layers:
    print(lay.name)
    print(lay.get_weights())
    



