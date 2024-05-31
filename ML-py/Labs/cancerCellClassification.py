import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("cell_samples.csv")
data.head()

ax = data[data['Class'] == 4][0:50].plot(kind="scatter", y="UnifSize", x="Clump",
        color="red", label="malignant");
data[data['Class'] == 2][0:50].plot(kind="scatter", x="Clump", y="UnifSize", 
        label ="benign" ,color ='DarkBlue',ax=ax);
plt.show()

data.dtypes

data = data[pd.to_numeric(data['BareNuc'], errors="coerce").notnull()]
data['BareNuc'] = data['BareNuc'].astype('int')
data.dtypes


feature_df = data[['Clump', 'UnifSize', 'UnifShape', 'MargAdh',
                   'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl',
                   'Mit']]
X = np.asarray(feature_df)
y = np.asarray(data['Class'])

y[0:5]
X[0:5]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4, test_size=0.2)
print('Train set:', X_train.shape, y_train.shape)
print('Test set', X_test, y_test)

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

result = clf.predict(X_test)
result[0:5]

from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
cnf_matrix = confusion_matrix(y_test, result, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, result))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')


from sklearn.metrics import f1_score
print("f1 score for that model is -> " + f1_score(y_test, result, average='weighted') )

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        random_state=5,test_size=0.25)

from sklearn import svm
model2 = svm.SVC(kernel="linear")
model2.fit(X_train, y_train)

model2result = model2.predict(X_test)
model2result[0:5]

from sklearn.metrics import f1_score
f1_score(y_test, model2result, average='weighted') 
# 0.9639038982104676 -> 0.9709401709401709