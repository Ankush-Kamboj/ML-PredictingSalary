import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np 
from sklearn import linear_model

# loading the data
dataset = pd.read_csv("Salaries.csv")
# dataset = ds[['YearsExperience', 'Salary']]

# creating train and test datasets
msk = np.random.rand(len(dataset)) < 0.8
train = dataset[msk]
test = dataset[~msk]

# training the model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['YearsExperience']])
train_y = np.asanyarray(train[['Salary']])
regr.fit(train_x, train_y)

# Making Predictions
test_x = np.asanyarray(test[['YearsExperience']])
test_y = np.asanyarray(test[['Salary']])
test_y_pred = regr.predict(test_x)

# making new Predictions
experience = 2
salary = regr.predict([[experience]])
print(salary[0][0])