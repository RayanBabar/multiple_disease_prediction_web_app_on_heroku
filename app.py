# Importing dependencies
import joblib
import streamlit as st
from streamlit_option_menu import option_menu


# Loading models
diabetes_model_and_scaler = joblib.load("diabetes_prediction_model.joblib")
diabetes_model = diabetes_model_and_scaler["model"]
diabetes_scaler = diabetes_model_and_scaler["scaler"]

heart_model = joblib.load("heart_disease_pred_model.joblib")

parkinson_model_and_scaler = joblib.load("parkinson_pred_model.joblib")
parkinson_model = parkinson_model_and_scaler["model"]
parkinson_scaler = parkinson_model_and_scaler["scaler"]

breast_model_and_scaler = joblib.load("breast_cancer_pred_model.joblib")
breast_model = breast_model_and_scaler["model"]
breast_scaler = breast_model_and_scaler["scaler"]


# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction",
        [
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Parkinsons Prediction",
            "Breast Cancer Prediction",
        ],
        icons=["activity", "heart", "person", "bookmark"],
        default_index=0,
    )

# Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    st.write(
        "This app predicts whether or not a patient has diabetes based on the patient's medical history."
    )

    col1, col2, col3, = st.columns(3)

    with col1:
        Pregnancies = st.text_input("No. of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("Blood Pressure Level")
    with col1:
        SkinThickness = st.text_input("Skin Thickness")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:
        bmi = st.text_input("BMI value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    with col2:
        Age = st.text_input("Age")

    if st.button("Diabetes Test Result"):
        diab_data_std = diabetes_scaler.transform(
            [
                [
                    Pregnancies,
                    Glucose,
                    BloodPressure,
                    SkinThickness,
                    Insulin,
                    bmi,
                    DiabetesPedigreeFunction,
                    Age,
                ]
            ]
        )
        diabetes_pred = diabetes_model.predict(diab_data_std)
        if diabetes_pred[0] == 1:
            st.error("Patient has diabetes!")
        elif diabetes_pred[0] == 0:
            st.success("Patient does not have diabetes!")


# Heart Disease Prediction Page
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    st.write(
        "This app predicts whether or not a patient has heart disease based on the patient's medical history."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age")
    with col2:
        sex = st.text_input("Sex")
    with col3:
        cp = st.text_input("Chest Pain types")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        chol = st.text_input("Serum Cholestoral in mg/dl")
    with col3:
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl")
    with col1:
        restecg = st.text_input("Resting Electrocardiographic results")
    with col2:
        thalach = st.text_input("Maximum Heart Rate achieved")
    with col3:
        exang = st.text_input("Exercise Induced Angina")
    with col1:
        oldpeak = st.text_input("ST depression induced by exercise")
    with col2:
        slope = st.text_input("Slope of the peak exercise ST segment")
    with col3:
        ca = st.text_input("Major vessels colored by flourosopy")
    with col1:
        thal = st.text_input(
            "thal: (0 = normal; 1 = fixed defect; 2 = reversable defect)"
        )

    if st.button("Heart Disease Test Result"):
        heart_prediction = heart_model.predict(
            [
                [
                    age,
                    sex,
                    cp,
                    trestbps,
                    chol,
                    fbs,
                    restecg,
                    thalach,
                    exang,
                    oldpeak,
                    slope,
                    ca,
                    thal,
                ]
            ]
        )
        if heart_prediction[0] == 1:
            st.error("The person is having heart disease!")
        elif heart_prediction[0] == 0:
            st.success("The person does not have any heart disease!")


# Parkinsons Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinsons Prediction")
    st.write(
        "This app predicts whether or not a patient has Parkinsons based on the patient's medical history."
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input("MDVP Fo(Hz)")
    with col2:
        fhi = st.text_input("MDVP Fhi(Hz)")
    with col3:
        flo = st.text_input("MDVP Flo(Hz)")
    with col4:
        Jitter_percent = st.text_input("MDVP Jitter(%)")
    with col5:
        Jitter_Abs = st.text_input("MDVP Jitter(Abs)")
    with col1:
        RAP = st.text_input("MDVP RAP")
    with col2:
        PPQ = st.text_input("MDVP PPQ")
    with col3:
        DDP = st.text_input("Jitter DDP")
    with col4:
        Shimmer = st.text_input("MDVP Shimmer")
    with col5:
        Shimmer_dB = st.text_input("MDVP Shimmer(dB)")
    with col1:
        APQ3 = st.text_input("Shimmer APQ3")
    with col2:
        APQ5 = st.text_input("Shimmer APQ5")
    with col3:
        APQ = st.text_input("MDVP APQ")
    with col4:
        DDA = st.text_input("Shimmer DDA")
    with col5:
        NHR = st.text_input("NHR")
    with col1:
        HNR = st.text_input("HNR")
    with col2:
        RPDE = st.text_input("RPDE")
    with col3:
        DFA = st.text_input("DFA")
    with col4:
        spread1 = st.text_input("spread1")
    with col5:
        spread2 = st.text_input("spread2")
    with col1:
        D2 = st.text_input("D2")
    with col2:
        PPE = st.text_input("PPE")

    if st.button("Parkinson's Test Result"):
        parkinson_data_std = parkinson_scaler.transform(
            [
                [
                    fo,
                    fhi,
                    flo,
                    Jitter_percent,
                    Jitter_Abs,
                    RAP,
                    PPQ,
                    DDP,
                    Shimmer,
                    Shimmer_dB,
                    APQ3,
                    APQ5,
                    APQ,
                    DDA,
                    NHR,
                    HNR,
                    RPDE,
                    DFA,
                    spread1,
                    spread2,
                    D2,
                    PPE,
                ]
            ]
        )
        parkinsons_pred = parkinson_model.predict(parkinson_data_std)

        if parkinsons_pred[0] == 1:
            st.error("The person has Parkinson's disease!")
        elif parkinsons_pred[0] == 0:
            st.success("The person does not have Parkinson's disease!")


# Breast Cancer Prediction Page
if selected == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction")
    st.write(
        "This app predicts whether or not a patient has breast cancer based on the patient's medical history."
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        mean_radius = st.text_input("mean radius".capitalize())
    with col2:
        mean_texture = st.text_input("mean texture".capitalize())
    with col3:
        mean_perimeter = st.text_input("mean perimeter".capitalize())
    with col4:
        mean_area = st.text_input("mean area".capitalize())
    with col5:
        mean_smoothness = st.text_input("mean smoothness".capitalize())
    with col1:
        mean_compactness = st.text_input("mean compactness".capitalize())
    with col2:
        mean_concavity = st.text_input("mean concavity".capitalize())
    with col3:
        mean_concave_points = st.text_input("mean concave points".capitalize())
    with col4:
        mean_symmetry = st.text_input("mean symmetry".capitalize())
    with col5:
        mean_fractal_dimension = st.text_input("mean fractal dim".capitalize())
    with col1:
        radius_error = st.text_input("radius error".capitalize())
    with col2:
        texture_error = st.text_input("texture error".capitalize())
    with col3:
        perimeter_error = st.text_input("perimeter error".capitalize())
    with col4:
        area_error = st.text_input("area error".capitalize())
    with col5:
        smoothness_error = st.text_input("smoothness error".capitalize())
    with col1:
        compactness_error = st.text_input("compactness error".capitalize())
    with col2:
        concavity_error = st.text_input("concavity error".capitalize())
    with col3:
        concave_points_error = st.text_input("concave points error".capitalize())
    with col4:
        symmetry_error = st.text_input("symmetry error".capitalize())
    with col5:
        fractal_dimension_error = st.text_input("fractal dim error".capitalize())
    with col1:
        worst_radius = st.text_input("worst radius".capitalize())
    with col2:
        worst_texture = st.text_input("worst texture".capitalize())
    with col3:
        worst_perimeter = st.text_input("worst perimeter".capitalize())
    with col4:
        worst_area = st.text_input("worst area".capitalize())
    with col5:
        worst_smoothness = st.text_input("worst smoothness".capitalize())
    with col1:
        worst_compactness = st.text_input("worst compactness".capitalize())
    with col2:
        worst_concavity = st.text_input("worst concavity".capitalize())
    with col3:
        worst_concave_points = st.text_input("worst concave points".capitalize())
    with col4:
        worst_symmetry = st.text_input("worst symmetry".capitalize())
    with col5:
        worst_fractal_dimension = st.text_input("worst fractal dim".capitalize())

    if st.button("Breast Cancer Test Result"):
        breast_data_std = breast_scaler.transform(
            [
                [
                    mean_radius,
                    mean_texture,
                    mean_perimeter,
                    mean_area,
                    mean_smoothness,
                    mean_compactness,
                    mean_concavity,
                    mean_concave_points,
                    mean_symmetry,
                    mean_fractal_dimension,
                    radius_error,
                    texture_error,
                    perimeter_error,
                    area_error,
                    smoothness_error,
                    compactness_error,
                    concavity_error,
                    concave_points_error,
                    symmetry_error,
                    fractal_dimension_error,
                    worst_radius,
                    worst_texture,
                    worst_perimeter,
                    worst_area,
                    worst_smoothness,
                    worst_compactness,
                    worst_concavity,
                    worst_concave_points,
                    worst_symmetry,
                    worst_fractal_dimension,
                ]
            ]
        )
        breast_pred = breast_model.predict(breast_data_std)
        if breast_pred[0] == 1:
            st.error("The person has breast cancer!")
        elif breast_pred[0] == 0:
            st.success("The person does not have breast cancer!")
