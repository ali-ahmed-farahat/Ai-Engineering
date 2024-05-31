import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


df = pd.read_csv("FuelConsumptionCo2.csv")

# take a look at the dataset
#df.head()
#print("hello world")
#df.describe()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]
# cdf.head(9)
# viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
# viz.hist()
# plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color = 'red')
plt.xlabel('Fuel Consumption')
plt.ylabel('Co2 Emissions')


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = 'purple')
plt.xlabel('Engine Size')
plt.ylabel('Co2 Emissions (target)')

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color = 'green')
plt.xlabel('Cylinders of the car')
plt.ylabel('Co2 emissions of the car')


mask = np.random.rand(len(df)) < 0.8
training = cdf[mask]
test = cdf[~mask]


plt.scatter(training.ENGINESIZE, training.CO2EMISSIONS, color = 'blue')
plt.xlabel('Engine Size')
plt.ylabel('Co2 Emissions')
plt.show()

from sklearn import linear_model

model = linear_model.LinearRegression()
train_x = np.asanyarray(training[['ENGINESIZE']])
train_y = np.asanyarray(training[['CO2EMISSIONS']])
model.fit(train_x, train_y)

print('the coefficient is', model.coef_)
print('the intercept is', model.intercept_)

plt.scatter(training.ENGINESIZE, training.CO2EMISSIONS)
plt.plot(train_x, model.intercept_[0] + model.coef_[0][0] * train_x, '-r')
plt.xlabel("Engine Size")
plt.ylabel("co2 emissions")
plt.show()


from sklearn.metrics import r2_score

test_x = np.asanyarray(training[['ENGINESIZE']])
test_y = np.asanyarray(training[['CO2EMISSIONS']])
prediction = model.predict(test_x)

MAE = np.mean(np.absolute(prediction - test_y))
MSE = np.mean(prediction - test_y) ** 2
R_squared = r2_score(prediction, test_y)

print('MSE = ', MSE)
print('MAE = ', MAE)
print('R2 = ', R_squared)


from sklearn import linear_model

model_2 = linear_model.LinearRegression()
x = np.asanyarray(training[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(training[['CO2EMISSIONS']])
model_2.fit(x,y)
print('Coefficients: ', model_2.coef_)

prediction = model_2.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])

MAE = np.mean(np.absolute(prediction - y))
MSE = np.mean((prediction - y) **2)
score = model_2.score(x,y)

print('MSE = ', MSE)
print('MAE = ', MAE)
print('Variance score = ', score)



from sklearn import linear_model

model3 = linear_model.LinearRegression()
x = np.asanyarray(training[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(training[['CO2EMISSIONS']])
model3.fit(x,y)
print('Coefficients: ', model3.coef_)

prediction = model3.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])

MAE = np.mean(np.absolute(prediction - y))
MSE = np.mean((prediction - y) **2)
score = model3.score(x,y)

print('MSE = ', MSE)
print('MAE = ', MAE)
print('Variance score = ', score)