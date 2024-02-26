import joblib
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
#import tensorflow as tf
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier

model_filename = 'model/diabaties_model_og.joblib'
model = joblib.load(model_filename)

# Load the models
svm_model = joblib.load('model/svm_model.joblib')
dt_model = joblib.load('model/decision_tree_model.joblib')
rf_model = joblib.load('model/random_forest_model.joblib')
lr_model = joblib.load('model/logistic_regression_model.joblib')

def prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    # Convert input values to numeric
    a = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    a = [float(val) for val in a]  # Convert to float
    
    # Make prediction
    prediction = model.predict([a])[0] 

    if int(prediction) == 1:
        print('Patient is Diabetic')
        return 'Patient is Diabetic'
    else:
        print('Patient is non Diabetic')
        return 'Patient is Non Diabetic'

def predict_disease_lr(input_data,symptoms):
    data = {}
    for i in input_data:
        for j in symptoms:
            if j != i:
                data[j] = 0
            else:
                data[j] = 1

    data = pd.DataFrame([data])

    # Predict using the trained Logistic Regression model
    prediction = lr_model.predict(data)[0]
    prob_scores = lr_model.predict_proba(data)[0]

    unique_labels = lr_model.classes_
    prob_dict = {label: prob for label, prob in zip(unique_labels, prob_scores)}

    return prediction, prob_dict

def predict_disease_rf(input_data,symptoms):
    data = {}
    for i in input_data:
        for j in symptoms:
            if j != i:
                data[j] = 0
            else:
                data[j] = 1

    data = pd.DataFrame([data])

    # Predict using the trained Random Forest model
    prediction = rf_model.predict(data)[0]
    prob_scores = rf_model.predict_proba(data)[0]

    unique_labels = rf_model.classes_
    prob_dict = {label: prob for label, prob in zip(unique_labels, prob_scores)}

def predict_disease_dt(input_data,symptoms):
    data = {}
    for i in input_data:
        for j in symptoms:
            if j != i:
                data[j] = 0
            else:
                data[j] = 1

    data = pd.DataFrame([data])

    # Predict using the trained Decision Tree model
    prediction = dt_model.predict(data)[0]

    # Decision Trees don't have built-in predict_proba method
    prob_dict = {label: 1.0 if label == prediction else 0.0 for label in dt_model.classes_}

    return prediction, prob_dict

def predict_disease_svm(input_data,symptoms):
    data = {}
    for i in input_data:
        for j in symptoms:
            if j != i:
                data[j] = 0
            else:
                data[j] = 1

    data = pd.DataFrame([data])

    # Predict using the trained SVM model
    prediction = svm_model.predict(data)[0]
    prob_scores = svm_model.predict_proba(data)[0]

    unique_labels = svm_model.classes_
    prob_dict = {label: prob for label, prob in zip(unique_labels, prob_scores)}

    return prediction, prob_dict




# Disease Prediction using Symtoms

symptoms_data = pd.read_csv('dataset/symptoms_data.csv').to_dict()['Disease']

label_encoder = joblib.load('model/label_encoder.joblib')

tuned_models = joblib.load('model/tuned_models.joblib')

def prediction_system_disease(dat):
    dat = dat.strip()  # Removes any leading or trailing spaces
    symptoms_list = dat.split(',')  # Splits the string into a list using comma as delimiter
    symptoms_list = [symptom.strip() for symptom in symptoms_list]
    print(symptoms_list,'aaaaaaaaaaaaaaaaaa')
    preparing_data = []
    for i,j in symptoms_data.items():
        if j in symptoms_list:
            preparing_data.append(1)
        else:
            preparing_data.append(0)
    
    preparing_data = np.array(preparing_data) 
    print(preparing_data,'bbbbbbbbbbbbbb') # Convert to NumPy array
    prediction_by_model = {}
    
    for model_name, model_instance in tuned_models.items():
        if model_name != 'VotingClassifier':
            # Assuming preparing_data is formatted correctly for predictions
            y_pred = model_instance.predict([preparing_data])  
            y_pred = label_encoder.inverse_transform(y_pred)[0]
            prediction_by_model[model_name] = y_pred
    
    dis = list(prediction_by_model.values())
    return max(set(dis), key = dis.count),dis  # Return the predictions
