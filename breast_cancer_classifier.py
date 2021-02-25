import pandas as pd
import numpy as np

# read the csv file
data = pd.read_csv('breast-cancer-wisconsin.csv')

# we need to remove missing values

# replace '?' values with np.nan
data.replace('?', np.nan, inplace=True)

# now remove nan values via dropna function
new = pd.DataFrame
new = data.dropna(axis=0)

# features
x = new.iloc[:, 1:10].values
# target
y = new.iloc[:, 10:].values

# splitting data into training and testing
from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# building the model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train, Y_train)
y_pred = knn.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# accuracy score
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))