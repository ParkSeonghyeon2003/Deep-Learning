from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load dataset
dataframe = pd.read_csv("data/liver.csv")
dataset = dataframe.values
X = dataset[:, 1:7].astype(float)
Y = dataset[:, 0]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)

# Devide train, test
train_X, test_X, train_y, test_y = train_test_split(
    X, dummy_y, test_size=0.4, random_state=321
)

# Standardization
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

# define model (DNN structure)
epochs = 200
batch_size = 10

model = Sequential()
model.add(Input(shape=(6,)))
model.add(Dense(10, activation='relu'))
model.add(Dropout(rate=0.15))
model.add(Dense(10, activation='relu'))
model.add(Dropout(rate=0.15))
model.add(Dense(8, activation='relu'))
model.add(Dropout(rate=0.05))
model.add(Dense(2, activation='softmax'))

model.summary() # show model structure

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# model fitting (learning)
disp = model.fit(
    train_X, train_y,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(test_X, test_y)
)

# Test model
pred = model.predict(test_X)
print(pred)
y_classes = [np.argmax(y, axis=None, out=None) for y in pred]
print(y_classes)

# model performance
score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# summarize history for accuracy
plt.plot(disp.history['accuracy'])
plt.plot(disp.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# model weights
for lay in model.layers:
    print(lay.name)
    print(lay.get_weights())
