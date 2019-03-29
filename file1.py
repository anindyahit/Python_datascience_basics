import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

iris= load_iris()
X= iris.data
Y= iris.target

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, random_state=42, test_size=0.5)


# Fitting XGBoost to the Training set

from xgboost import XGBClassifier

model= XGBClassifier()

model.fit(X_train,Y_train)

prediction = model.predict(X_test)

print(accuracy_score(prediction,Y_test))


#pickle file
import pickle
with open(r'C:\Users\achowdhury\Desktop\Demo\rf.pkl','wb') as model_pkl:
    pickle.dump(model, model_pkl)


