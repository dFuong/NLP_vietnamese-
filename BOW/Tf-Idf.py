import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm, metrics
from sklearn.metrics import accuracy_score

# read data
np.random.seed(500)
data = pd.read_csv('./name.csv',names=["name","sexual"])

# create list
doc = []

for word in data['name']:
    word = str(word)
    array = []

    for i in range(0,26):
        array.append([0,0])
        # array = np.array([0,0])
    for i in range(0,len(word)):
        if ( (ord(word[i]) >= ord('a')) and (ord(word[i]) <= ord('z')) ):
            array[ord(word[i]) - ord('a')][1] = array[ord(word[i]) - ord('a')][1] + i
            array[ord(word[i]) - ord('a')][0] = array[ord(word[i]) - ord('a')][0] + 1

    result = []

    for item in array:
        result.append(item[0])
        result.append(item[1])

    doc.append(result)
    # doc.append(temp)

# print(doc)

# split the dataset into training and test datasets
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(doc,data['sexual'],test_size=0.25)

# encoding
Encoder = LabelEncoder()
Train_Y= Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


# LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(Train_X, Train_Y)
predictions_Logistic= clf.predict(Test_X)
# accuracy: 80.12135221034383
print("LogisticRegression Accuracy Score -> ",accuracy_score(predictions_Logistic, Test_Y)*100)


# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# SVM.fit(Train_X,Train_Y)
# predictions_SVM = SVM.predict(Test_X)
# # accuracy:80.65876914186651
# print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

# from sklearn.svm import SVC
# clf = SVC(kernel='rbf', gamma= 2, C = 200)
# clf.fit(Train_X,Train_Y)
# predictions_SVC = clf.predict(Test_X)
# # 77
# print("SVC accuracy score -> ", accuracy_score(predictions_SVC,Test_Y)*100)
