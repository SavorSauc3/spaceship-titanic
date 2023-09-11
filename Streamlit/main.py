# Import Dependencies(Install these if you haven't yet...)
import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

st.title('Spaceship Titanic Prediction Tutorial')

# Sets values for numerical features + categorical features
numerical_features = ['Room_Number', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
all_features = ['Room_Number', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Pipeline for numerical features
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical features
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(handle_unknown='ignore'))
])

# Combine them into a preprocessor
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])

# Input Formatting
class Person:
    def __init__(self, age, room_num, room_serv, food_ct, shop_mall, spa, vr, home_planet, cryo, dest, vip, deck, side):
        self.age = age
        self.room_num = room_num
        self.room_serv = room_serv
        self.food_ct = food_ct
        self.shop_mall = shop_mall
        self.spa = spa
        self.vr = vr
        self.home_planet = home_planet
        self.cryo = cryo
        self.dest = dest
        self.vip = vip
        self.deck = deck
        self.side = side

# Instantiate Person
profile = Person(None, None, None, None, None, None, None, None, None, None, None, None, None)

# Input Form
with st.form(key='person_form'):
    profile.age = st.slider('Pick your age', 0, 100)
    profile.room_num = st.number_input('Pick a room number', 0, 500)
    profile.room_serv = st.number_input('How much did you spend on Room Service', 0, 10000)
    profile.food_ct = st.number_input('How much did you spend at the Food Court', 0, 10000)
    profile.shop_mall = st.number_input('How much did you spend at the Shopping Mall', 0, 10000)
    profile.spa = st.number_input('How much did you spend at the Spa', 0, 10000)
    profile.vr = st.number_input('How much did you spend at the VRDeck', 0, 10000)
    profile.home_planet = st.radio('What is your Home Planet', ['Earth', 'Mars', 'Europa'])
    profile.cryo = st.toggle('CryoSleep')
    profile.dest = st.radio('Where are you going', ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22'])
    profile.vip = st.toggle('VIP')
    profile.deck = st.radio('What deck was your room on', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    profile.side = st.radio('What side was your room on P = Port, S = Starboard', ['S', 'P'])
    st.form_submit_button('Submit Person')

# Put this here to make sure I had all the features
all_features = ['Room_Number', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
data = {
    'Age' : profile.age,
    'Room_Number' : profile.room_num,
    'RoomService' : profile.room_serv,
    'ShoppingMall' : profile.shop_mall,
    'Spa' : profile.spa,
    'VRDeck' : profile.vr,
    'HomePlanet' : profile.home_planet,
    'FoodCourt' : profile.food_ct,
    'CryoSleep' : profile.cryo,
    'Destination' : profile.dest,
    'VIP' : profile.vip,
    'Deck' : profile.deck,
    'Side' : profile.side
}

# Show data being passed to model
framed_data = pd.DataFrame(data, index=[0])
st.table(framed_data)

# Transform the data
preprocessor = joblib.load('preprocessor.sav')
model_data = preprocessor.transform(framed_data)

gbm_model_1 = joblib.load('gbm_model_1.sav')
gbm_model_2 = joblib.load('gbm_model_2.sav')
gbm_model_3 = joblib.load('gbm_model_3.sav')
meta_model = joblib.load('meta_model.sav')

# Predict using these models, and show final predictions

gbm_1_predictions = gbm_model_1.predict(model_data)
gbm_2_predictions = gbm_model_2.predict(model_data)
gbm_3_predictions = gbm_model_3.predict(model_data)

stacked_preds = np.column_stack((gbm_1_predictions, gbm_2_predictions, gbm_3_predictions))
final_prediction = meta_model.predict(stacked_preds)
st.write(f'Prediction was {final_prediction}')

