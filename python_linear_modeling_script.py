#!/usr/bin/env python

# Python Linear Modeling Assignment
# BGSP 7030 SU2023

# Import the libraries needed: pandas, sklearn, matplotlib

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys 

# Printing 0th argument value 
print(sys.argv[0]) 

print("Running Linear Modelling of Data Python Script") 

# Used read_csv() to read regrex1.csv

filename = "regrex1.csv"
print("loading filename {}".format(filename))
print()
# Call the read.csv function from the pandas library to read regrex1.csv

dataset = pd.read_csv(filename)
dataset.describe()
print(dataset)

# Fitting Linear Regression to the dataset

model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])

# Scatter Plot of Original Data

plt.scatter(dataset[['x']], dataset[['y']], color = 'green')
plt.title('x vs y')
plt.xlabel('x')
plt.ylabel('y')


# Visualizing the Linear Regression results

plt.scatter(dataset[['x']], dataset[['y']], color = 'green')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'cyan')
plt.title('regrex1 linear regression')
plt.xlabel('x')
plt.ylabel('y')





