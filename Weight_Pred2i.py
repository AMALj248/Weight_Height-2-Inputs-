import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataSet
dataset = pd.read_csv("weight-height.csv")
dataset.info()
dataset.describe()
dataset.isnull().sum()

#in this model i use both inputs gender and height

#Convert Gender to number
# Replace directly in dataFrame
dataset['Gender'].replace('Female',0, inplace=True)
dataset['Gender'].replace('Male',1, inplace=True)
X = dataset[["Gender","Height"]]
#X = dataset.iloc[:, 1].values
Y = dataset.iloc[:,2].values
X=X/10
Y=Y/10
# checking input was correctly passsed
print(X)
print(Y)




# SPLITTING THE DATA
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


#we will use the LinearRegression to train our model

from sklearn.linear_model import LinearRegression
ln_rg = LinearRegression()
ln_rg.fit(X_train, Y_train)

#Predict test set values with the test set
ln_rg_pred = ln_rg.predict(X_test)
Y_pred= ln_rg_pred

#plotting the scatter plot  between y_actual and y_predicited
plt.scatter(Y_test, Y_pred , c='green')
plt.xlabel;("Input Weight")
plt.ylabel("predicted Weight X10")
plt.title(" True vs Predicted value : Linear Regression X10")
plt.show()

#Result from the LINEAR REGRESSION MODEL
#mean squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, Y_pred)
print(" Mean Square Error : ", mse)
#Mean absolute error
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred- Y_test))))


