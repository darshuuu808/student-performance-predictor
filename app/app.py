import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model('app/model.keras')
scaler = joblib.load('app/scaler.pkl')
feature_names = joblib.load('app/feature_names.pkl')

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.markdown("Predict whether a student will **Pass or Fail** based on their profile using a Neural Network trained on real student data.")

# Sidebar
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown("**Model:** Neural Network (TensorFlow/Keras)")
    st.markdown("**Dataset:** UCI Student Performance (Portuguese)")
    st.markdown("**Task:** Binary Classification")
    st.markdown("**Test Accuracy:** 90.82%")
    st.markdown("**Val Accuracy:** 87.63%")
    st.divider()
    st.markdown("**Pass Threshold:** G3 ≥ 10")
    st.markdown("**Neural Network beats Logistic Regression by 1.02%**")

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

# Encode inputs
encode_binary = lambda x: 1 if x == "yes" else 0

input_dict = {
    'school': 0,
    'sex': 0,
    'age': age,
    'address': 0,
    'famsize': 0 if famsize == "LE3" else 1,
    'Pstatus': 1 if Pstatus == "T" else 0,
    'Medu': Medu,
    'Fedu': Fedu,
    'Mjob': 0,
    'Fjob': 0,
    'reason': 0,
    'guardian': ["mother","father","other"].index(guardian),
    'traveltime': 1,
    'studytime': studytime,
    'failures': failures,
    'schoolsup': encode_binary(schoolsup),
    'famsup': encode_binary(famsup),
    'paid': encode_binary(paid),
    'activities': 0,
    'nursery': 1,
    'higher': encode_binary(higher),
    'internet': encode_binary(internet),
    'romantic': encode_binary(romantic),
    'famrel': famrel,
    'freetime': freetime,
    'goout': goout,
    'Dalc': Dalc,
    'Walc': Walc,
    'health': health,
    'absences': absences
}

# Align with training feature order
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
            st.markdown(f"The student is likely to **pass**.")
        else:
            st.error("## ❌ FAIL")
            st.markdown(f"The student is likely to **fail**.")

    with col_res2:
        st.metric("Pass Probability", f"{prob*100:.1f}%")
        st.progress(float(prob))

    with col_res3:
        st.metric("Fail Probability", f"{(1-prob)*100:.1f}%")
        st.progress(float(1-prob))