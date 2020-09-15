# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:07:40 2020

@author: parth
"""

import pickle
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_iris


#loading data
data= load_iris()
X=data.data
y=data.target


#Spliting data 
X_train,X_test,y_train,y_test = train_test_split(X,y)


#loading the model
model=RandomForestClassifier()


#training the classifier 
model.fit(X_train, y_train)


#checking the accuracy
predicted_result=model.predict(X_test)
print(accuracy_score(y_test,predicted_result))


#saving the model
pickle.dump(model,open('random_iris_model.pkl', 'wb'))