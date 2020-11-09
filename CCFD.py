import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#Test Valid imports
#print('Python: {}'.format(sys.version))
#print('Numpy: {}'.format(numpy.__version__))
#print('Pandas: {}'.format(pandas.__version__))
#print('Matplotlib: {}'.format(matplotlib.__version__))
#print('Seaborn: {}'.format(seaborn.__version__))
#print('Scipy: {}'.format(scipy.__version__))
#print('Sklearn: {}'.format(sklearn.__version__))

#Loading Data from CSV
data = pd.read_csv('creditcard.csv')

#explore dataset
#print(data.columns)
#print(data.shape)
#print(data.describe())

#sample of data
data = data.sample(frac = 0.1, random_state = 1)
#print(data.shape)

#Plot histogram of paramaters
#data.hist(figsize = (20,20))
#plt.show()

#Determine number of frad cases in data
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud) / float(len(valid))
#print(outlier_fraction)

#print('Fraud Cases: {}'.format(len(fraud)))
#print('Valid Cases: {}'.format(len(valid)))

#Correlation matrix
#corrmat = data.corr()
#fig = plt.figure(figsize = (12,9))

#sns.heatmap(corrmat,vmax = .8,square = True)
#plt.show()

#Get all the columns from the DataFrame
columns = data.columns.tolist()

#Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ['Class']]
#Store the variable we will be predicting on
target = "Class"
X = data[columns]
Y = data[target]
#Print the shapes of X and Y
print(X.shape)
print(Y.shape)

#define a random state
state = 1

#define outlier detection methods
classifiers = { 
    "Isolation Forest": IsolationForest(max_samples = len(X),
                                        contamination = outlier_fraction,
                                        random_state = state),
    "Local Outlier Factor": LocalOutlierFactor(
    n_neighbors = 20,
    contamination = outlier_fraction)
}

#fit the model
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    #reshapee the prediction values to 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()

    #run classfication matrics
    print('{}: {}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))


