import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/AABA_2006-01-01_to_2018-01-01.csv', engine='python')
print(df.head())

# 전처리
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df.plot()
plt.show()

# train = ~2017-06-30
# test = 2017-07-01~
split_date = pd.Timestamp('2017-06-30')

train = df.loc[:split_date, ['Close']]
test = df.loc[split_date:, ['Close']]

ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])
plt.show()

# 데이터 정규화
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], 
                            index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], 
                            index=test.index)
print(train_sc_df.head())

for s in range(1, 31):
    train_sc_df['shift_{}'.format(s)] = \
                train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = \
                test_sc_df['Scaled'].shift(s)

print(train_sc_df.head(31))

# NA 포함 행 제거
X_train = train_sc_df.dropna().drop('Scaled', axis=1)
y_train = train_sc_df.dropna()[['Scaled']]

X_test = test_sc_df.dropna().drop('Scaled', axis=1)
y_test = test_sc_df.dropna()[['Scaled']]

# 데이터 순서 변경
columns = X_train.columns[30::-1]
X_train = X_train[columns]
X_test = X_test[columns]

print(X_train.head())
print(X_test.head())

# dataframe to ndarray
X_train = X_train.values
X_test = X_test.values

y_train = y_train.values
y_test = y_test.values
print(X_train.shape)
print(X_train)
print(y_train.shape)
print(y_train)

# 최종 데이터셋 구성
X_train_t = X_train.reshape((X_train.shape[0], 30, 1))
X_test_t = X_test.reshape((X_test.shape[0], 30, 1))

print("Final DATASET")
print(X_train_t.shape)
print(X_train_t)
print(y_train)

# LSTM 모델 구축
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
# from keras.callbacks import EarlyStopping

K.clear_session()
model = Sequential()
model.add(LSTM(20, input_shape=(30, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

model.fit(X_train_t, y_train, epochs=100, batch_size=30, verbose=1)

y_pred = model.predict(X_test_t)
print(y_pred)

from sklearn.metrics import mean_squared_error
print('Mean squared error: {0:.2f}'.\
        format(mean_squared_error(y_test, y_pred)))

plt.figure()
plt.plot(y_pred)
plt.plot(y_test)
plt.legend(['predict', 'actual'])
plt.show()
