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

train = df.loc[:split_date, ['Open', 'High', 'Low', 'Close']]
test = df.loc[split_date:, ['Open', 'High', 'Low', 'Close']]

ax = train.plot()
test.plot(ax=ax)
plt.legend(['train_Open', 'train_High', 'train_Low', 'train_Close',
            'test_Open', 'test_High', 'test_Low', 'test_Close'])
plt.show()

# 데이터 정규화
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

train_sc_df = pd.DataFrame(train_sc, columns=['Open', 'High', 'Low', 'Close'], 
                            index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Open', 'High', 'Low', 'Close'], 
                            index=test.index)
print(train_sc_df.head())

# shift 연산 - 각 feature별로
for s in range(1, 31):
    for col in ['Open', 'High', 'Low', 'Close']:
        train_sc_df['{}_{}'.format(col, s)] = train_sc_df[col].shift(s)
        test_sc_df['{}_{}'.format(col, s)] = test_sc_df[col].shift(s)

print(train_sc_df.head(31))

# y_train, y_test는 Close만 사용
y_train = train_sc_df.dropna()[['Close']]
y_test = test_sc_df.dropna()[['Close']]

# X_train, X_test는 shift된 컬럼들만 사용
feature_columns = []
for s in range(1, 31):
    for col in ['Open', 'High', 'Low', 'Close']:
        feature_columns.append('{}_{}'.format(col, s))

X_train = train_sc_df.dropna()[feature_columns]
X_test = test_sc_df.dropna()[feature_columns]

# 데이터 순서 변경 (과거 -> 현재)
# shift_30 -> shift_1 순서로
columns_reversed = []
for s in range(30, 0, -1):
    for col in ['Open', 'High', 'Low', 'Close']:
        columns_reversed.append('{}_{}'.format(col, s))

X_train = X_train[columns_reversed]
X_test = X_test[columns_reversed]

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
# (samples, timesteps, features) = (samples, 30, 4)
X_train_t = X_train.reshape((X_train.shape[0], 30, 4))
X_test_t = X_test.reshape((X_test.shape[0], 30, 4))

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
model.add(LSTM(20, input_shape=(30, 4)))  # 4개 feature 사용
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
