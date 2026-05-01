import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/darshuuu808/student-performance-predictor/main/data/student-mat.csv"

@st.cache_resource
def load_or_train_model():
    df = pd.read_csv(DATA_URL, sep=',')
    df['pass'] = (df['G3'] >= 10).astype(int)
    df = df.drop(columns=['G1', 'G2'])

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=['G3', 'pass'])
    y = df['pass']
    feature_names = list(X.columns)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, batch_size=16, callbacks=callbacks, verbose=0)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    return model, scaler, feature_names, round(test_acc * 100, 2)

st.title("🎓 Student Performance Predictor")
st.markdown("Predict whether a student will **Pass or Fail** based on their profile using a Neural Network trained on real student data.")

with st.spinner("⏳ Loading model... (first load takes ~30 seconds)"):
    model, scaler, feature_names, test_acc = load_or_train_model()

# Sidebar
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown("**Model:** Neural Network (TensorFlow/Keras)")
    st.markdown("**Dataset:** UCI Student Performance")
    st.markdown("**Task:** Binary Classification")
    st.markdown(f"**Test Accuracy:** {test_acc}%")
    st.markdown("**Pass Threshold:** G3 ≥ 10")

st.divider()
st.subheader("📋 Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📚 Academic**")
    studytime = st.selectbox("Weekly Study Time",
        options=[1,2,3,4],
        format_func=lambda x: {1:"<2 hrs",2:"2-5 hrs",3:"5-10 hrs",4:">10 hrs"}[x])
    failures = st.slider("Past Class Failures", 0, 4, 0)
    absences = st.slider("Number of Absences", 0, 93, 5)
    schoolsup = st.selectbox("Extra School Support", ["yes", "no"])
    famsup = st.selectbox("Family Educational Support", ["yes", "no"])
    paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])

with col2:
    st.markdown("**👨‍👩‍👦 Family**")
    age = st.slider("Age", 15, 22, 17)
    famsize = st.selectbox("Family Size", ["LE3", "GT3"])
    Pstatus = st.selectbox("Parents' Cohabitation", ["T", "A"],
        format_func=lambda x: {"T":"Together","A":"Apart"}[x])
    Medu = st.slider("Mother's Education (0-4)", 0, 4, 2)
    Fedu = st.slider("Father's Education (0-4)", 0, 4, 2)
    famrel = st.slider("Family Relationship Quality (1-5)", 1, 5, 3)
    guardian = st.selectbox("Guardian", ["mother", "father", "other"])

with col3:
    st.markdown("**🎮 Lifestyle**")
    freetime = st.slider("Free Time After School (1-5)", 1, 5, 3)
    goout = st.slider("Going Out with Friends (1-5)", 1, 5, 3)
    Dalc = st.slider("Workday Alcohol (1-5)", 1, 5, 1)
    Walc = st.slider("Weekend Alcohol (1-5)", 1, 5, 1)
    health = st.slider("Health Status (1-5)", 1, 5, 3)
    internet = st.selectbox("Internet Access", ["yes", "no"])
    romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])

st.divider()

encode_binary = lambda x: 1 if x == "yes" else 0

input_dict = {
    'school': 0, 'sex': 0, 'age': age, 'address': 0,
    'famsize': 0 if famsize == "LE3" else 1,
    'Pstatus': 1 if Pstatus == "T" else 0,
    'Medu': Medu, 'Fedu': Fedu,
    'Mjob': 0, 'Fjob': 0, 'reason': 0,
    'guardian': ["mother","father","other"].index(guardian),
    'traveltime': 1, 'studytime': studytime, 'failures': failures,
    'schoolsup': encode_binary(schoolsup),
    'famsup': encode_binary(famsup),
    'paid': encode_binary(paid),
    'activities': 0, 'nursery': 1,
    'higher': encode_binary(higher),
    'internet': encode_binary(internet),
    'romantic': encode_binary(romantic),
    'famrel': famrel, 'freetime': freetime, 'goout': goout,
    'Dalc': Dalc, 'Walc': Walc, 'health': health, 'absences': absences
}

input_array = np.array([[input_dict[f] for f in feature_names]])
input_scaled = scaler.transform(input_array)

if st.button("🔍 Predict", use_container_width=True):
    prob = model.predict(input_scaled)[0][0]
    pred = int(prob >= 0.5)

    st.divider()
    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        if pred == 1:
            st.success("## ✅ PASS")
            st.markdown("The student is likely to **pass**.")
        else:
            st.error("## ❌ FAIL")
            st.markdown("The student is likely to **fail**.")

    with col_res2:
        st.metric("Pass Probability", f"{prob*100:.1f}%")
        st.progress(float(prob))

    with col_res3:
        st.metric("Fail Probability", f"{(1-prob)*100:.1f}%")
        st.progress(float(1-prob))