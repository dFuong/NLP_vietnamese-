import pandas as pd
import numpy as np
import re
import collections
from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, svm, metrics, preprocessing
from sklearn.metrics import accuracy_score

# load data
np.random.seed(500)
data = pd.read_csv('./name1.csv',names=["name","sexual"])

# split word into character
data['name'] = [list(word) for word in data['name']]
# print(data['name'].head())

# pre_processing
documents = []

for word in data['name']:
     document = re.sub(r'\W', ' ', str(word))
     # document = re.sub(r"[']", "",str(word))
     # document = re.sub(r'^b\s+', '', document)
     document = document.lower()
     document = document.split()
     document = ' '.join(document)
     documents.append(document)
# print(documents)

# split the dataset into training and validation datasets
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(documents,data['sexual'],test_size=0.3)

# encoding
Encoder = LabelEncoder()
Train_Y= Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

#Tf_idf
Tfidf_vect = TfidfVectorizer(analyzer='word',max_features=5000, tokenizer=lambda word: word)
Tfidf_vect.fit_transform(documents)
Tfidf_vect.get_feature_names()
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# # LogisticRegression
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression()
# clf.fit(Train_X_Tfidf, Train_Y)
# predictions_Logistic= clf.predict(Test_X_Tfidf)
# # accuracy: 76.36
# print("LogisticRegression Accuracy Score -> ",accuracy_score(predictions_Logistic, Test_Y)*100)

# SVM
# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# SVM.fit(Train_X_Tfidf,Train_Y)
# predictions_SVM = SVM.predict(Test_X_Tfidf)
# # accuracy:76.86506789945102
# print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

# SVC
# from sklearn.svm import SVC
# clf = SVC(kernel='rbf', gamma= 3, C = 200)
# clf.fit(Train_X_Tfidf,Train_Y)
# predictions_SVC = clf.predict(Test_X_Tfidf)
# print("SVC accuracy score -> ", accuracy_score(predictions_SVC,Test_Y)*100)

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()
# clf.fit(Train_X_Tfidf, Train_Y)
# predictions_Logistic= clf.predict(Test_X_Tfidf)
# # accuracy: 87.80121352210344
# print("LogisticRegression Accuracy Score -> ",accuracy_score(predictions_Logistic, Test_Y)*100)