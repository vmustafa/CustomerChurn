import streamlit as sl
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle as pkl
import pandas as pd
import numpy as np

## load the trained model
model = tf.keras.models.load_model('model.h5')

## load the encoders and scaler
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pkl.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pkl.load(file)

with open('scalar.pkl','rb') as file:
    scalar=pkl.load(file)

## streamlit 
sl.title("Customer Churn Prediction")

## user inputs
geography = sl.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = sl.selectbox("Gender", label_encoder_gender.classes_)
age = sl.slider("Age", 18, 92)
balance = sl.number_input("Balance")
credit_score = sl.number_input("Credit Score")
estimated_salary = sl.number_input("Estimated Salary")
tenure = sl.slider("Tenure", 0, 10)
num_of_products = sl.slider("Number of Products", 1, 4)
has_credit_card = sl.selectbox("Has Credit Card", [0,1])
is_active_member = sl.selectbox("Is Active Member", [0,1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scalar.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]
print(prediction_proba)

sl.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    sl.write('The customer is likely to churn.')
else:
    sl.write('The customer is not likely to churn.')


