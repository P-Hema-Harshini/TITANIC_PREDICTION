import streamlit as st
import joblib
import pandas as pd

def load_model():
    model = joblib.load('titanic_model.pkl')
    return model

def load_features():
    features = joblib.load('model_features.pkl')
    return features


st.title("Titanic Survival Prediction App")

pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 30)
sibsp = st.number_input('Number of Siblings/Spouses', min_value=0, max_value=8, value=0)
parch = st.number_input('Number of Parents/Children', min_value=0, max_value=6, value=0)
fare = st.number_input('Fare', min_value=0.0, max_value=500.0, value=50.0)
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex_male': [1 if sex == 'male' else 0],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0],
})

model = load_model()
features = load_features()

input_data = input_data.reindex(columns=features, fill_value=0)

if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.success("The passenger survived.")
    else:
        st.error("The passenger did not survive.")
