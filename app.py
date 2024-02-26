import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from helper import *
from sklearn.preprocessing import StandardScaler
import pandas as pd

from helper import symptoms_data
symptoms = list(symptoms_data.values())
#print(symptoms,'<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>','hhhhhhhhhhhhhhhhhhhhhhhhh')
symptoms =[str(i).strip() for i in symptoms]
#dataset = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\Diabetic\\dataset\\dataset.csv')

st.sidebar.title('Diabetes ANALYSIS')
sidebar_data = st.sidebar.radio(
    'Select an Option',
    ('Diabetes Prediction','Disease Prediction','Non Diabetes Analysis')
)

if sidebar_data == 'Diabetes Prediction':
    st.subheader('Diabetes Prediction')
    col1 ,col2 =st.columns(2)
    with col1:
        Pregnancies = st.text_input("Pregnancies count:")
    with col2:
        Glucose = st.text_input("Glucose count:")
    with col1:
        BloodPressure = st.text_input("BloodPressure count:")
    with col2:
        SkinThickness = st.text_input("SkinThickness count:")
    with col1:
        Insulin = st.text_input("Insulin count:")
    with col2:
        BMI = st.text_input("BMI count:")
    with col1:
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction count:")
    with col2:
        Age = st.text_input("Age count:")

    if st.button("Check Prediction",key='prediction_button', type="primary"):
        if all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            #st.write(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
            predicted = prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
            
            st.write(predicted)
    
        else:
            st.warning("Please fill in all input values before submitting.")



elif sidebar_data == 'Disease Prediction':
    st.subheader('Disease Prediction')
    st.write(symptoms)
    # Define columns for layout
    with st.form("my_form"):
        # Add a text input field
        user_name = st.text_input("Enter your name:")
        user_id = st.text_input("Enter your userid:")
        dat = st.text_area('Enter Symptoms U are facing')
        print(user_name,user_id,dat,'<<<<<<<<<<<>>>>>>>>>>>>>>>')
        # Add a button to submit the form
        submit_button = st.form_submit_button(label="Submit")

    # Process the form submission
        if submit_button:
            disease,dis = prediction_system_disease(dat)
            st.write(f"Hello, {user_name}!")
            st.write(f'U are suffering from {disease}')