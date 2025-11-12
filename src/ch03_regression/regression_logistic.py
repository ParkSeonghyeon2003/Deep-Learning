# logistic Regression Example

from sklearn import datasets
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
iris_X, iris_y = datasets.load_iris(return_X_y=True)
print(iris_X.shape)   # (150, 4)

# Split the data into training/testing sets
train_X, test_X, train_y, test_y = \
    train_test_split(iris_X, iris_y, test_size=0.3,\
                     random_state=1234) 

# Define learning model
model =  LogisticRegression()

# Train the model using the training sets
model.fit(train_X, train_y)

# Make predictions using the testing set
pred_y = model.predict(test_X)
print(pred_y)

# model evaluation: accuracy #############

acc = accuracy_score(test_y, pred_y)
print('Accuracy : {0:3f}'.format(acc))
