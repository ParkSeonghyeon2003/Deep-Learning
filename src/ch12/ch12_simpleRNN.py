# 앞쪽 4개의 숫자가 주어졌을 때 그 다음에 올 숫자를 예측
# ref:  https://chonchony.tistory.com/entry/Tensorflow-Keras-SimpleRNN-%EC%8B%A4%EC%8A%B5

import keras
import numpy as np

# 학습 데이터 생성
X = []
Y = []

for i in range(6):
    # [0,1,2,3], [1,2,3,4] 같은 정수의 시퀀스 생성
    lst = list(range(i,i+4))

    # 위에서 구한 시퀀스의 숫자들을 각각 10으로 나눈 다음 저장.
    # 각 타임스텝(계층)에 숫자가 하나씩 들어가기 때문에 분리해서 배열에 저장.
    X.append(list(map(lambda c: [c/10], lst)))

    # 정답에 해당하는 4, 5 등의 정수를 역시 위처럼 10으로 나눠서 저장합니다.
    Y.append((i+4)/10)
    
X = np.array(X)
Y = np.array(Y)

for i in range(len(X)):
    print(X[i], Y[i])

# 시퀀스 예측 모델 정의

model = keras.Sequential([
    keras.layers.SimpleRNN(units=10, return_sequences=False, input_shape=[4,1]),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit(X, Y, epochs=100, verbose=0)

# 학습된 시퀀스에 대한 예측 결과
print(model.predict(X))

# 학습되지 않은 시퀀스에 대한 예측 결과
print(model.predict(np.array([[[0.6],[0.7],[0.8],[0.9]]])))
print(model.predict(np.array([[[-0.1],[0.0],[0.1],[0.2]]])))
