import streamlit as st
import numpy as np
import joblib

# ---------------------
# تحميل الموديل (Random Forest)
# ---------------------
model_path = "model.pkl"



try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.title("🎓 College Placement Prediction (Random Forest)")

# ---------------------
# إدخال القيم من اليوزر
# ---------------------
IQ = st.slider("IQ", 50, 160, 100)
Prev_Sem_Result = st.slider("Previous Semester Result (%)", 0, 100, 70)
CGPA = st.slider("CGPA", 0.0, 10.0, 7.0)
Academic_Performance = st.slider("Academic Performance (0-10)", 0, 10, 7)
Internship_Experience = st.number_input("Internships Done", min_value=0, max_value=5, value=0)
Extra_Curricular_Score = st.slider("Extra Curricular Score (0-10)", 0, 10, 5)
Communication_Skills = st.slider("Communication Skills (0-10)", 0, 10, 6)
Projects_Completed = st.number_input("Projects Completed", min_value=0, max_value=20, value=2)

# ---------------------
# تجهيز الداتا
# ---------------------
input_data = np.array([[float(IQ), float(Prev_Sem_Result), float(CGPA), 
                        float(Academic_Performance), float(Internship_Experience), 
                        float(Extra_Curricular_Score), float(Communication_Skills), 
                        float(Projects_Completed)]])

# ---------------------
# التنبؤ + الاحتمالية
# ---------------------
if st.button("Predict Placement"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        placed_prob = round(proba[1] * 100, 2)

        if prediction == 1:
            st.success(f"Yes ✅ (Probability: {placed_prob}%)")
        else:
            st.warning(f"No ❌ (Probability: {placed_prob}%)")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")




