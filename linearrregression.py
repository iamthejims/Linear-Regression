# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:37:50 2022

@author: Jims
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


db = pd.read_csv('C:\\Users\\Jims\\Desktop\\Salary_Data.csv')

#divide the dataset inti test and train

x = db.iloc[:,:-1].values

y = db.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =  train_test_split(x,y, test_size = .2 , random_state = 0)

from sklearn.linear_model import LinearRegression

sr = LinearRegression()
sr.fit(x_train,y_train)

y_predict = sr.predict(x_test)

plt.scatter(x_train,y_train, color='red')
plt.plot(x_train, sr.predict(x_train))
plt.title('Salary vs Experience')
plt.xlabel('exp')
plt.ylabel('salary')
plt.show()

plt.scatter(x_test,y_test, color='red')
plt.plot(x_train, sr.predict(x_train))
plt.title('Salary vs Experience')
plt.xlabel('exp')
plt.ylabel('salary')
plt.show()
