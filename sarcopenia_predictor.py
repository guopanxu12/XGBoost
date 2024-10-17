import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('xgb_best_model.pkl')

# Define feature names
feature_names = [
    "Age (年龄)", "BMI (体重指数)", "WC (腰围)", "DBP (舒张压)", "HGB (血红蛋白)"
]

# Streamlit user interface
st.title("中老年心血管病患者肌少症风险预测 (Sarcopenia Risk Prediction for Middle-aged and Elderly Cardiovascular Patients)")

# Age: categorical selection
age = st.selectbox("Age (年龄):", options=[1, 2, 3], format_func=lambda x: "45-64岁" if x == 1 else "65-74岁" if x == 2 else "≥75岁")

# BMI: categorical selection
bmi = st.selectbox("BMI (体重指数):", options=[1, 2, 3], format_func=lambda x: "<18.5 kg/m2" if x == 1 else "18.5-23.9 kg/m2" if x == 2 else "≥24 kg/m2")

# WC: numerical input
wc = st.number_input("WC (腰围) (cm):", min_value=40, max_value=160, value=80)

# DBP: numerical input
dbp = st.number_input("DBP (舒张压) (mmHg):", min_value=40, max_value=140, value=80)

# HGB: numerical input
hgb = st.number_input("HGB (血红蛋白) (g/L):", min_value=60, max_value=200, value=130)

# Process inputs and make predictions
feature_values = [age, bmi, wc, dbp, hgb]
features = np.array([feature_values], dtype=float)

if st.button("预测 (Predict)"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**预测类别 (Predicted Class):** {predicted_class}")
    st.write(f"**预测概率 (Prediction Probabilities):** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            "根据模型预测，您可能存在较高的肌少症发病风险。\n"
            f"模型预测的发病概率为 {probability:.1f}%。\n"
            "建议您尽快就医，以进行更详细的诊断和采取适当的治疗措施。\n\n"
            "According to the model, you may have a high risk of developing sarcopenia.\n"
            f"The predicted probability of developing sarcopenia is {probability:.1f}%.\n"
            "It is recommended that you see a doctor as soon as possible for a more detailed diagnosis and appropriate treatment."
        )
    else:
        advice = (
            "根据模型预测，您的肌少症风险较低。\n"
            f"模型预测的无肌少症概率为 {probability:.1f}%。\n"
            "建议您继续保持健康的生活方式，并定期观察健康状况。如有任何异常症状，请及时就医。\n\n"
            "According to the model, your risk of sarcopenia is low.\n"
            f"The predicted probability of not having sarcopenia is {probability:.1f}%.\n"
            "It is recommended that you maintain a healthy lifestyle and monitor your health regularly. If you experience any symptoms, please see a doctor promptly."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
