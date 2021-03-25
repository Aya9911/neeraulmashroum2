#Hasnaa Mohamed Reda Abo Ghaba CS sec 2
#Fatma Saeid CS sec 3
#Aya Elkordy IT sec 1

import pandas as pd
dataset = pd.read_csv("D:\Home\mushrooms.csv")


from sklearn.preprocessing  import LabelEncoder
lbl = LabelEncoder()

for col in dataset.columns:
    dataset[col] = lbl.fit_transform(dataset[col])
        

import matplotlib.pyplot as plt

dataset.hist(bins=30, figsize=(14, 16))
plt.show()


X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(9),max_iter=500)

mlp.fit(X_train,y_train)

y_pred = mlp.predict(X_test)


from sklearn.metrics import confusion_matrix,classification_report
print()

print('artificial neural network')
print()
print (confusion_matrix(y_test, y_pred))

print(classification_report(y_test,y_pred))
print()

print("this is kneighbors")
print()
print()

from sklearn.neigaahbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
clf = classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import  confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))

print()
print()
print()
print()


print("this is neural network after applying PCA")
print()
print()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_new = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(9),max_iter=500)

mlp.fit(X_train,y_train)

y_pred = mlp.predict(X_test)

from sklearn.metrics import  confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))

plt.figure(figsize=(10,9))
colors = ['r','b']
colors2 = ['c','b']

for i in range(len(x_new)):
    plt.scatter(x_new[i][0], x_new[i][1], c=colors[y[i]], s=5)

for j in range(len(y_test)):
    plt.scatter(X_test[:,0], X_test[:,1],c=colors2[y_pred[j]], marker='<')  

plt.show()















