#!/usr/bin/env python

# # Python Linear Modeling Assignment
# BGSP 7030 SU2023


# Import the libraries needed: pandas, sklearn, matplotlib - os and sys as well 
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os 
import sys 

# variable for filename
if len(sys.argv) < 2:
    print("Missing filename")
    sys.exit(-1)

filename = sys.argv[1]

# Split filename into base and ext for ease of naming 
base,ext = os.path.splitext(filename)
print(base)
print(ext)

# Print out current process, which is loading the dataset of the file 
print("Loading dataset from {}".format(filename))
print()
      
# Call the read.csv function from the pandas library to read file
dataset = pd.read_csv(filename)
dataset.describe()
print(dataset)

# Fitting Linear Regression to the dataset
model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])

#Adjusted R-Squared 
model.score(dataset[['x']], dataset[['y']])

      
# Scatter Plot of Original Data   
plt.scatter(dataset[['x']], dataset[['y']], color = 'green')
plt.title("y vs x for {}".format(base))
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("{} scatter.png".format(filename))


# Visualizing the Linear Regression results
plt.scatter(dataset[['x']], dataset[['y']], color = 'green')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'cyan')
plt.title("Linear regression for {}".format(base))
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("{} linear regression".format(base))



