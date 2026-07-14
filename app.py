import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# ============================================
# Custom CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .danger {
        background: #ffebee;
        border-left: 5px solid #f44336;
    }
    .safe {
        background: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Load Model
# ============================================
@st.cache_resource
def load_model():
    model_path = 'heart_disease_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

# ============================================
# Main App
# ============================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1> Heart Disease Predictor</h1>
        <p>ระบบทำนายความเสี่ยงโรคหัวใจ</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ ไม่พบไฟล์โมเดล heart_disease_model.pkl")
        st.info("กรุณารัน train_model.py ก่อน")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 เกี่ยวกับระบบ")
        st.info("โมเดล Decision Tree สำหรับทำนายโรคหัวใจ")
    
    # Input Form
    st.markdown("### 📝 กรอกข้อมูลสุขภาพ")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("อายุ (ปี)", min_value=20, max_value=100, value=50)
        sex = st.selectbox("เพศ", ["ชาย", "หญิง"])
        sex_value = 1 if sex == "ชาย" else 0
        
        cp = st.selectbox("อาการเจ็บหน้าอก", 
                         [1, 2, 3, 4],
                         format_func=lambda x: {
                             1: "Typical Angina",
                             2: "Atypical Angina",
                             3: "Non-anginal Pain",
                             4: "Asymptomatic"
                         }[x])
        
        trestbps = st.number_input("ความดันโลหิต (mm Hg)", 
                                   min_value=80, max_value=200, value=120)
    
    with col2:
        chol = st.number_input("คอเลสเตอรอล (mg/dl)", 
                              min_value=0, max_value=600, value=200)
        
        fbs = st.selectbox("น้ำตาลในเลือด > 120 mg/dl", 
                          ["ไม่ใช่", "ใช่"])
        fbs_value = 1 if fbs == "ใช่" else 0
        
        restecg = st.selectbox("ผล ECG", 
                              [0, 1, 2],
                              format_func=lambda x: {
                                  0: "Normal",
                                  1: "ST-T abnormality",
                                  2: "LV hypertrophy"
                              }[x])
        
        thalach = st.number_input("อัตราการเต้นหัวใจสูงสุด", 
                                 min_value=60, max_value=220, value=150)
    
    with col3:
        exang = st.selectbox("เจ็บหน้าอกขณะออกกำลังกาย", 
                            ["ไม่มี", "มี"])
        exang_value = 1 if exang == "มี" else 0
        
        oldpeak = st.number_input("Oldpeak", 
                                 min_value=-3.0, max_value=10.0, value=1.0)
        
        slope = st.selectbox("ความชัน ST segment", 
                            [1, 2, 3],
                            format_func=lambda x: {
                                1: "Upsloping",
                                2: "Flat",
                                3: "Downsloping"
                            }[x])
    
    # Predict Button
    if st.button(" ทำนายผล", use_container_width=True):
        # Create input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex_value],
            'ChestPainType': [cp],
            'RestingBP': [trestbps],
            'Cholesterol': [chol],
            'FastingBS': [fbs_value],
            'RestingECG': [restecg],
            'MaxHR': [thalach],
            'ExerciseAngina': [exang_value],
            'Oldpeak': [oldpeak],
            'ST_Slope': [slope]
        })
        
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display Result
        st.markdown("---")
        
        if prediction == 1:
            st.markdown("""
            <div class="result-card danger">
                <h2>⚠️ มีความเสี่ยงเป็นโรคหัวใจ</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card safe">
                <h2>✅ ไม่พบความเสี่ยงโรคหัวใจ</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Show probability
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ความน่าจะเป็นไม่เป็นโรค", f"{probability[0]*100:.1f}%")
        with col2:
            st.metric("ความน่าจะเป็นเป็นโรค", f"{probability[1]*100:.1f}%")
        
        st.progress(float(probability[1]))

if __name__ == "__main__":
    main()