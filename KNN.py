import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib as mpl
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/Users/Usman Khan/Desktop/AI-Project/stack-overflow-2018-developer-survey/survey_results_public.csv")
# length of Data 
print("Length of Data", len(dataset))
dataset.drop(['CompanySize'], 1, inplace= True)
dataset.fillna(0, inplace = True)


# converting Non numeric data to numeric means Encoding
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
print(dataset.head())

# Selecting Data 
X = dataset.iloc[:, 1:12].values
Y = dataset.iloc[:, 12].values

# Training and Spliting of Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.2)

#Scaling of Data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Appling KNNClassifier Model andd Train
classifier = KNeighborsClassifier(n_neighbors = 11)
classifier.fit(X_train, Y_train)

#Making Prediction for Testing of Data
y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(Y_test, y_pred)
#plt.matshow(cm)
print(cm)

# Accuracy
print("Accuracy", accuracy_score(Y_test, y_pred))

#Grapgh

plt.plot(Y_test, y_pred)
plt.show()
print("KNN")