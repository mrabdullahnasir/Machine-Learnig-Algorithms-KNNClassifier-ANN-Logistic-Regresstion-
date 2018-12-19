import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/Usman Khan/Desktop/Bilal/mushrooms.csv')



# Encoder
def handle_non_numeric_data(dataset):
    columns = dataset.columns.values
    for column in columns:
        text_digit_values = {}
        def convert_to_int(val):
            return text_digit_values[val]
        
        if dataset[column].dtype != np.int64 and dataset[column].dtype != np.float64:
            column_contents = dataset[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_values:
                    text_digit_values[unique]= x
                    x+=1
            dataset[column] = list(map(convert_to_int, dataset[column]))
    return dataset

dataset = handle_non_numeric_data(dataset)
X = dataset.iloc[:, 1:12].values
Y = dataset.iloc[:,12].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

sq = Sequential()
sq.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

sq.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

sq.compile(optimizer = 'adam',loss='binary_crossentropy',metrics = ['accuracy'])
sq.fit(X_train, Y_train, batch_size = 10, nb_epoch=100)

y_pred = sq.predict(X_test)
y_pred = (y_pred > 0.5)
new_pred = sq.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))

new_pred = (new_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(Y_test, y_pred)
cm
plt.matshow(cm)
# plt.plot(Y_test, y_pred)
# plt.show()
# print("ANN")