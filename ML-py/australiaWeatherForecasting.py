import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics


data = pd.read_csv("Weather_Data.csv")

weather_data = pd.get_dummies(data, columns=['RainTommorow','RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])

weather_data.replace(['Yes', 'No'], [1, 0], inplace=True)

weather_data.drop('Date',axis=1,inplace=True)

weather_data_processed = weather_data.astype(float)

features = weather_data_processed.drop(columns='RainTomorrow', axis=1)
Y = weather_data_processed['RainTomorrow']


#Use the `train_test_split` function to split the `features` and `Y` dataframes with a `test_size` of `0.2` and the `random_state` set to `10`
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=10)

#Q2) Create and train a Linear Regression model called LinearReg using the training data (`x_train`, `y_train`)
LinearReg = LinearRegression()
LinearReg.fit(x_train, y_train)
#Q3) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`
predictions = LinearReg.predict(x_test)
#Q4) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function.
LinearRegression_MAE = metrics.mean_absolute_error(y_test, predictions)
LinearRegression_MSE = metrics.mean_squared_error(y_test, predictions)
LinearRegression_R2 = metrics.r2_score(y_test, predictions)
#Q5) Show the MAE, MSE, and R2 in a tabular format using data frame for the linear model.
results = {
    'Metric': ['Mean Absolute Error', 'Mean Squared Error', 'R^2 Score'],
    'Value': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]
}

results_df = pd.DataFrame(results)
print(results_df)


#Q6) Create and train a KNN model called KNN using the training data (`x_train`, `y_train`) with the `n_neighbors` parameter set to `4`
KNN = KNeighborsClassifier(n_neighbors=4).fit(x_train, y_train)
#Q7) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`
predictions = KNN.predict(x_test)
#Q8) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function
KNN_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
KNN_JaccardIndex = metrics.jaccard_score(y_test, predictions)
KNN_F1_Score = metrics.f1_score(y_test, predictions)

#Q9) Create and train a Decision Tree model called Tree using the training data (`x_train`, `y_train`)
Tree = DecisionTreeClassifier()
Tree.fit(x_train,y_train)
#Q10) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`
predictions = Tree.predict(x_test)
#Q11) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function
Tree_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
Tree_JaccardIndex = metrics.jaccard_score(y_test, predictions)
Tree_F1_Score = metrics.f1_score(y_test, predictions)


#Q12) Use the `train_test_split` function to split the `features` and `Y` dataframes with a `test_size` of `0.2` and the `random_state` set to `1`
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)

# Q13) Create and train a LogisticRegression model called LR using the training data (`x_train`, `y_train`) with the `solver` parameter set to `liblinear`
LR = LogisticRegression(solver='liblinear').fit(x_train,y_train)

#Q14) Now, use the `predict` and `predict_proba` methods on the testing data (`x_test`) and save it as 2 arrays `predictions` and `predict_proba`
predict_proba = LR.predict_proba(x_test)
predictions = LR.predict(x_test)

#Q15) Using the `predictions`, `predict_proba` and the `y_test` dataframe calculate the value for each metric using the appropriate function
LR_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
LR_JaccardIndex = metrics.jaccard_score(y_test, predictions)
LR_F1_Score = metrics.f1_score(y_test, predictions)
LR_Log_Loss = metrics.log_loss(y_test, predict_proba)

#Q16) Create and train a SVM model called SVM using the training data (`x_train`, `y_train`)
SVM = svm.SVC(kernel="linear")
SVM.fit(x_train, y_train)

#Q17) Now use the `predict` method on the testing data (`x_test`) and save it to the array `predictions`
predictions = SVM.predict(x_test)

#Q18) Using the `predictions` and the `y_test` dataframe calculate the value for each metric using the appropriate function
SVM_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
SVM_JaccardIndex = metrics.jaccard_score(y_test, predictions)
SVM_F1_Score = metrics.f1_score(y_test, predictions)

#Q19) Show the Accuracy,Jaccard Index,F1-Score and LogLoss in a tabular format using data frame for all of the above models

KNNpredictions = KNN.predict(x_test)
Treepredictions = Tree.predict(x_test)
LRpredictions = LR.predict(x_test)
SVMpredictions = SVM.predict(x_test)

models = {
    'KNN': (KNN, KNNpredictions),
    'Tree': (Tree, Treepredictions),
    'LR': (LR, LRpredictions),
    'SVM': (SVM, SVMpredictions)
}

# Assuming y_test is the same for all models
metrics_list = []

for model_name, (model, predictions) in models.items():
    accuracy = accuracy_score(y_test, predictions)
    jaccard = jaccard_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    if (model_name == "LR"):
        logloss = log_loss(y_test, model.predict_proba(x_test))
    else:
        logloss = None
    
    metrics_list.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Jaccard Index': jaccard,
        'F1 Score': f1,
        'Log Loss': logloss
    })

# Creating a DataFrame
Report = pd.DataFrame(metrics_list)

# Display the DataFrame
print(Report)
