#keras regression model for detecting the concrete strenght using a neural network built by keras (Regression model)



import keras
import tensorflow as tf
from keras import Sequential
import pandas as pd
import numpy as np
from keras import layers
data = pd.read_csv("concrete_data.csv")
data.head()

#checking if the data is clean or not
data.isnull().sum()

#splitting data into predictors and target columns
data_columns = data.columns
predictors = data[data_columns[data_columns != "Strength"]]
target = data['Strength']

#now lets normalize the data
predictors_norm = (predictors - predictors.mean()) / predictors.std()

#getting the number of columns
n_cols = predictors_norm.shape[1]


#let's create a function for creating a regression model
def regression_model():
    model = Sequential()
    
    model.add(layers.Dense(50, activation='relu', input_shape = (n_cols,)))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


#using the function to create the model and fit the model setting the num of iterations using epochs
#and using verbose to choose the way the results show, selecting showing 1  output line per epoch to visualize the process more easily

my_model = regression_model()
my_model.fit(predictors_norm, target, validation_split=0.3, epochs = 100, verbose = 2)
