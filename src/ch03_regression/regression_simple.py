# Simple Linear Regression Example
# cars dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# prepare dataset
cars = pd.read_csv('D:/data/cars.csv')
print(cars)
speed = cars['speed']
dist  = cars['dist']

speed = np.array(speed).reshape(50,1)
dist  = np.array(dist).reshape(50,1)

# Split the data into training/testing sets
train_X, test_X, train_y, test_y = \
    train_test_split(speed, dist, test_size=0.2, random_state=123) 

# Dfine learning method
model =  LinearRegression()

# Train the model using the training sets
model.fit(train_X, train_y)

# Make predictions using the testing set
pred_y = model.predict(test_X)
print(pred_y)

# prediction test
print(model.predict([[13]]))    # when speed=13 
print(model.predict([[20]]))    # when speed=20

# The coefficients & Intercept
print('Coefficients: {0:.2f}, Intercept {1:.3f}'\
      .format(model.coef_[0][0], model.intercept_[0]))

# The mean squared error
print('Mean squared error: {0:.2f}'.\
      format(mean_squared_error(test_y, pred_y))) 

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(test_y, pred_y))

# Plot outputs
plt.scatter(test_X, test_y,  color='black')
plt.plot(test_X, pred_y, color='blue', linewidth=3)

plt.xlabel('speed')
plt.ylabel('dist')

plt.show()