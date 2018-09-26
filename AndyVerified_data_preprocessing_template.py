# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import
dataset=pd.read_csv('Data.csv')

#separate into predictor and features
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3]


#handle missing numerical data
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])



#encode categorical data
# Var 1 in feature set. Use label encoder to set as 1,2,3.. then substitute with one hot encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x= LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder= OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()
#var2 response variable - python knows already that there is no ordering in response var, so label encoding is enough
labelencoder_y= LabelEncoder()
y = labelencoder_y.fit_transform(y)



# Splitting the dataset into the Training set and Test set
# random_state = 0 is neccessary for reproducibility
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



# Feature Scaling
# we should leave 'dummy variables' out if we want to leave interpretibility intact
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)

#for test set transform is enough. No fit needed because we want it scaled to resemble the training set
x_test = sc_x.transform(x_test)






