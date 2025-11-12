# Multiple Linear Regression Example

import pandas as ps
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the prestge dataset
df = ps.read_csv('D:\\OneDrive - 단국대학교\\2020_강의\\딥러닝클라우드_2020\\data\\prestige.csv')
print(df)
df_X = df[['education','women','prestige']]
df_y = df['income']

# Split the data into training/testing sets
train_X, test_X, train_y, test_y = \
    train_test_split(df_X, df_y, test_size=0.2) 

# Dfine learning model
model =  LinearRegression()

# Train the model using the training sets
model.fit(train_X, train_y)

# Make predictions using the testing set
pred_y = model.predict(test_X)
print(pred_y)

# The coefficients & Intercept
print('Coefficients: {0:.2f},{1:.2f},{2:.2f} Intercept {3:.3f}'\
      .format(model.coef_[0], model.coef_[1], model.coef_[2],\
              model.intercept_))

# The mean squared error
print('Mean squared error: {0:.2f}'.\
      format(mean_squared_error(test_y, pred_y))) 

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(test_y, pred_y))

# Test single data
my_test_x = np.array([11.44,8.13,54.1]).reshape(1,-1)
my_pred_y = model.predict(my_test_x)
print(my_pred_y)

