# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:24:12 2020

@author: parth
"""
#loading libraries
import pickle
from flask import Flask,request
import numpy as np
import pandas as pd
#loading the models
model=pickle.load(open('random_iris_model.pkl','rb'))
print(model.predict([[1,1,1,1]]))

#starting API
app=Flask(__name__)


@app.route('/')
def predict_iris():
    s_length=request.args.get("a")
    s_width=request.args.get("b")
    p_length=request.args.get("c")
    p_width=request.args.get("d")
    prediction=model.predict([[ s_length,s_width,p_length,p_width]])
    return str(prediction)


@app.route('/predict',methods=['POST'])
def predict_iris_file():
    input_data= pd.read_csv(request.files.get("input_file"),header=None)
    prediction = model.predict(input_data)
    return str(prediction)
    
    

if __name__ == "__main__":
    app.run()