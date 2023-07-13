import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

st.title(":blue[Diabetes Prediction System]")

df = pd.read_csv("diabetes.csv")

def prediction_train(df, input_data):
    # separating data and labels
    x = df.drop(columns='Outcome', axis=1)
    y = df['Outcome']
    # Data Standardization
    scaler = StandardScaler()
    scaler.fit(x)
    standardized_data = scaler.transform(x)
    X = standardized_data
    Y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    # training the model
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    # accuracy score
    X_train_pred = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_pred, y_train)
    # accuracy score
    X_test_pred = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_pred, y_test)

    inp = np.asarray(input_data)
    input_data_reshaped = inp.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)

    prediction = classifier.predict(std_data)
    return prediction

# Create a form using Streamlit form components
def user_report():
    # Form container
    st.markdown("#### Enter Patient Details")
    form_container = st.container()
    with form_container:
        box = st.container()
        with box:
            # Create nested container for columns
            col_container = st.container()
            with col_container:
                col1, col2 = st.columns(2)

            # Row 1
            with col1:
                pregnancies = st.number_input("Pregnancies", key="pregnancies", value=-1, step=1)
            with col2:
                glucose = st.number_input("Glucose", key="glucose", value=-1, step=1)

            # Row 2
            with col1:
                bp = st.number_input("BP", key="bp", value=-1, step=1)
            with col2:
                skinthickness = st.number_input("Skin Thickness", key="skinthickness", value=-1, step=1)

            # Row 3
            with col1:
                insulin = st.number_input("Insulin", key="insulin", value=-1, step=1)
            with col2:
                bmi = st.number_input("BMI", key="bmi", value=-1)

            # Row 4
            with col1:
                dpf = st.number_input("Diabetes Pedigree Function", key="dpf", value=-1)
            with col2:
                age = st.number_input("Age", key="age", value=-1, step=1)

            st.markdown("</div>", unsafe_allow_html=True)

    # Submit button
    if pregnancies == -1 or glucose == -1 or bp == -1 or skinthickness == -1 or insulin == -1 or bmi == -1 or dpf == -1 or age == -1:
        st.warning("Please fill in all the required fields.")
    else:
        submit_button = st.button("Submit", key="submit_button")

        # Process the form data on submission
        if submit_button:
            input_data = (pregnancies, glucose, bp, skinthickness, insulin, bmi, dpf, age)
            prediction = prediction_train(df, input_data)
            if prediction[0] == 0:
                st.markdown("##### :green[The person is not diabetic]")
            else:
                st.markdown("##### :red[The person is diabetic]")

# Run the form creation function
user_report()
