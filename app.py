import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
import tensorflow as tf
import pickle


# Loading the trained model 
model = tf.keras.models.load_model('model.h5')


# Load the scaler pickle, onehot
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('OneHot_encoder_geo.pkl','rb') as file:
    OneHot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)




# Streamlit app
st.title("Customer Churn Prediction")




# use inputs
geography =st.selectbox('Geography', OneHot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tuner',0,10)
num_of_products = st.slider('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit card',[0,1])
is_active_menber= st.selectbox('Is Active Member',[0,1])


# Prepare the inpute data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_menber],
    'EstimatedSalary':[estimated_salary]
})



# One-hot endode 'Geography'
geo_encoded = OneHot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=OneHot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction Churn
prediction =model.predict(input_data_scaled)
prediction_prob = prediction[0][0]


st.write(f"Churn Probability : {prediction_prob:.2f}")



if prediction_prob > 0.5:
    st.error("The customer is likely to churn.")
else:
    st.success("The customer is not likely to churn.")
