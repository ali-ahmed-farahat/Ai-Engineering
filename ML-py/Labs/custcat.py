#lab 2 summary and steps
# load the data
# select columns to use as X for training and testing
# select column  of result as y for training and testing
# normalize data to determine deviation using fit and transform by StandardScaler
# Train test split use X, y and testing size and random_state seeding number
# create a kN model by the KNeighborsClassifier and fit to x and y training DeprecationWarning
# predict the model by model.predict(x_train)
# evaluate accuracy by accuracy_score of testing data and training data
# plot all the k's accuracies possible to see which has the highest accuracy



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()
df['custcat'].value_counts()

df.hist(column='income', bins=50)
#selects columns and assign them to x and same with y
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]
y = df['custcat'].values
y[0:5]

#get the features of the X (mean & standard deviation)
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


from sklearn.neighbors import KNeighborsClassifier
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

#generating prediction on the classes of the customer classification
yhat = neigh.predict(X_test)
yhat[0:5]

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


k = 6
model = KNeighborsClassifier(n_neighbors= k).fit(X_train, y_train)
model

result_hat = model.predict(X_test)
result_hat[0:5]
from sklearn import metrics
print(metrics.accuracy_score(y_train, model.predict(X_train)))
print(metrics.accuracy_score(y_test, result_hat))



k = 4
neigh4 = KNeighborsClassifier(n_neighbors= k).fit(X_train, y_train)
neigh4

y_hat = neigh4.predict(X_test)
y_hat[0:5]

from sklearn import metrics
print(metrics.accuracy_score(y_train, neigh4.predict(X_train)))
print(metrics.accuracy_score(y_test, y_hat))

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 