# Simple Linear Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Tasmiya Anwer/Desktop/ml/Assignemnt-2-batch-3-master/Assignemnt-2-batch-3-master/size of heaad.csv')
X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Fitting Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# maping the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title('TRAINING SET - head size vs brain weight')
plt.xlabel('Head Size in cm^3')
plt.ylabel('Brain Weight in gm')
plt.show()

#maping the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title('TEST SET - Head Size vs Brain Weight ')
plt.xlabel('Head Size in cm^3')
plt.ylabel('Brain Weight in gm')
plt.show()