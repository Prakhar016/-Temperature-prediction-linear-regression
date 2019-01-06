#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Importing the dataset
dataset = pd.read_csv('nottem.csv') 
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#Spiltting the dataset into the training and testing set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#Fitting Sample Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression(normalize=True)
regressor.fit(X_train,y_train)

#predicting the test set result
y_pred=regressor.predict(X_test)

#Visulasing the training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Time  Vs Tempreature(Training Set)')
plt.xlabel('Time')
plt.ylabel('Temprature')
plt.show()

#Visulaize the Test set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Time  Vs Tempreature(testing set)')
plt.xlabel('Time')
plt.ylabel('Temprature')
plt.show()
  