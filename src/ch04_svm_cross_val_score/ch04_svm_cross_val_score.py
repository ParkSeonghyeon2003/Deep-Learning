# SVM cross_val_score Example

from sklearn import datasets
from sklearn import svm 
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the iris dataset
wine_X, wine_y = datasets.load_wine(return_X_y=True)

# Define learning model
model =  svm.SVC()

scores = cross_val_score(model, wine_X, wine_y, cv=5)

#kfold = KFold(n_splits=5, shuffle=True, random_state=123)
#scores = cross_val_score(model, wine_X, wine_y, cv=kfold)

print("fold acc", scores)
print("mean acc", np.mean(scores))
