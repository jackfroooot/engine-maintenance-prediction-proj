import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

   # Business can tune the threshold used to decide to go for maintenance or not
predict_threshold = 0.55

# Download and load the trained model
proj_name = "engine-maintenance-prediction-proj"
model_path = hf_hub_download(repo_id=f"jackfroooot/{proj_name}", filename="best_engine_maintenance_pred_model_v1.joblib", repo_type="model")
model = joblib.load(model_path)

# Streamlit UI
st.title("Engine Maintenance Prediction")
st.write("""
This application predicts whether an engine needs **maintenance** based on the sensor values. Please enter the sensor details below to get a maintenance prediction indication.
""")

# User input
Engine_rpm        = st.number_input("Engine rpm", min_value=0, max_value=5000, step=1, value=790)
Lub_oil_pressure  = st.number_input("Lub oil pressure", min_value=0.0, max_value=10.0, value=3.3)
Fuel_pressure     = st.number_input("Fuel pressure", min_value=0.0, max_value=25.0, value=6.65)
Coolant_pressure  = st.number_input("Coolant pressure", min_value=0.0, max_value=10.0, value=2.34)
lub_oil_temp      = st.number_input("Lub oil temp", min_value=0.0, max_value=100.0, value=77.0)
Coolant_temp      = st.number_input("Coolant temp", min_value=0.0, max_value=250.0, value=78.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
  'Engine rpm'        :  Engine_rpm      ,
  'Lub oil pressure'  :  Lub_oil_pressure,
  'Fuel pressure'     :  Fuel_pressure   ,
  'Coolant pressure'  :  Coolant_pressure,
  'lub oil temp'      :  lub_oil_temp    ,
  'Coolant temp'      :  Coolant_temp     
}])

# Predict button
if st.button("Predict Maintenance"):
    prediction = model.predict_proba(input_data)[:, 1][0]
    if prediction >= 0.85: risk = "🔴 High Risk"
    elif prediction >= predict_threshold:  risk = "🟠 Medium Risk"
    else:  risk = "🟢 Low Risk"

    st.subheader("Prediction Result:")
    st.success(f"Prediction (Risk={risk}, score={prediction:.2f}) : {'**Require Maintenance**' if prediction > predict_threshold else '**Does not need Maintenance**'}")
