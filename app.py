import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS for Beautiful Design
# ============================================
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #667eea;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #666;
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-card.danger {
        border-left: 5px solid #ff4b4b;
    }
    
    .result-card.safe {
        border-left: 5px solid #00c853;
    }
    
    .result-card h2 {
        margin: 0 0 1rem 0;
        font-size: 2rem;
    }
    
    .result-card.danger h2 {
        color: #ff4b4b;
    }
    
    .result-card.safe h2 {
        color: #00c853;
    }
    
    /* Input section */
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Load Model
# ============================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.pkl')
        return model
    except:
        st.error("❌ ไม่พบไฟล์โมเดล กรุณาอัปโหลดไฟล์ heart_disease_model.pkl")
        return None

# ============================================
# Main Application
# ============================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🫀 Heart Disease Predictor</h1>
        <p>ระบบทำนายความเสี่ยงโรคหัวใจด้วย Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # ============================================
    # Sidebar - Information
    # ============================================
    with st.sidebar:
        st.markdown("### 📊 เกี่ยวกับระบบ")
        st.info("""
        ระบบนี้ใช้โมเดล **Decision Tree** ในการทำนายความเสี่ยงโรคหัวใจ
        
        **Features ที่ใช้:**
        - อายุ
        - เพศ
        - ประเภทอาการเจ็บหน้าอก
        - ความดันโลหิต
        - คอเลสเตอรอล
        - น้ำตาลในเลือด
        - ผล ECG
        - อัตราการเต้นหัวใจสูงสุด
        - อาการเจ็บหน้าอกขณะออกกำลังกาย
        - ค่า Oldpeak
        - ความชัน ST segment
        """)
        
        st.markdown("### 📈 ข้อมูลโมเดล")
        st.success("""
        - **Algorithm:** Decision Tree
        - **Accuracy:** ~85%
        - **Training Data:** 900+ samples
        """)
        
        if st.button("🔄 รีเซ็ตข้อมูล"):
            st.experimental_rerun()
    
    # ============================================
    # Main Content - Input Form
    # ============================================
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 กรุณากรอกข้อมูลสุขภาพ")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("🎂 อายุ (ปี)", min_value=20, max_value=100, value=50, step=1)
        sex = st.selectbox("⚧️ เพศ", ["ชาย", "หญิง"])
        sex_value = 1 if sex == "ชาย" else 0
        
        cp = st.selectbox(
            "💔 ประเภทอาการเจ็บหน้าอก",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical Angina",
                2: "Atypical Angina",
                3: "Non-anginal Pain",
                4: "Asymptomatic"
            }[x]
        )
        
        trestbps = st.number_input("🩸 ความดันโลหิตขณะพัก (mm Hg)", 
                                   min_value=80, max_value=200, value=120, step=1)
    
    with col2:
        chol = st.number_input("🧪 คอเลสเตอรอล (mg/dl)", 
                              min_value=100, max_value=600, value=200, step=1)
        
        fbs = st.selectbox("🍬 น้ำตาลในเลือดอดอาหาร > 120 mg/dl", 
                          ["ไม่ใช่", "ใช่"])
        fbs_value = 1 if fbs == "ใช่" else 0
        
        restecg = st.selectbox(
            "📊 ผล Electrocardiographic",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T wave abnormality",
                2: "Left ventricular hypertrophy"
            }[x]
        )
        
        thalach = st.number_input("💓 อัตราการเต้นหัวใจสูงสุด", 
                                 min_value=60, max_value=220, value=150, step=1)
    
    with col3:
        exang = st.selectbox("🏃 อาการเจ็บหน้าอกขณะออกกำลังกาย", 
                            ["ไม่มี", "มี"])
        exang_value = 1 if exang == "มี" else 0
        
        oldpeak = st.number_input("📈 Oldpeak (ST depression)", 
                                 min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        slope = st.selectbox(
            "📉 ความชันของ ST segment",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping"
            }[x]
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================
    # Predict Button
    # ============================================
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 ทำนายผล", use_container_width=True)
    
    # ============================================
    # Prediction Logic
    # ============================================
    if predict_button:
        # Create input dataframe
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
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        
        if prediction == 1:
            # High risk
            st.markdown(f"""
            <div class="result-card danger">
                <h2>⚠️ มีความเสี่ยงเป็นโรคหัวใจ</h2>
                <p style="font-size: 1.2rem; color: #666;">
                    ความน่าจะเป็น: <strong>{probability[1]*100:.1f}%</strong>
                </p>
                <p style="color: #ff4b4b; margin-top: 1rem;">
                    🏥 แนะนำให้ปรึกษาแพทย์เพื่อตรวจเพิ่มเติม
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Low risk
            st.markdown(f"""
            <div class="result-card safe">
                <h2>✅ ไม่พบความเสี่ยงโรคหัวใจ</h2>
                <p style="font-size: 1.2rem; color: #666;">
                    ความน่าจะเป็น: <strong>{probability[0]*100:.1f}%</strong>
                </p>
                <p style="color: #00c853; margin-top: 1rem;">
                    🎉 รักษาสุขภาพต่อไปนะครับ!
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show probability breakdown
        st.markdown("### 📊 รายละเอียดความน่าจะเป็น")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="ไม่เป็นโรค",
                value=f"{probability[0]*100:.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                label="เป็นโรค",
                value=f"{probability[1]*100:.1f}%",
                delta=None
            )
        
        # Progress bar
        st.progress(probability[1])
        
        # Show input summary
        with st.expander("📋 ดูข้อมูลที่กรอก"):
            st.dataframe(input_data.T.rename(columns={0: 'ค่า'}))

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 1rem;">
    <p>🫀 Heart Disease Predictor | Powered by Machine Learning</p>
    <p style="font-size: 0.9rem; opacity: 0.8;">
        ⚠️ คำเตือน: ระบบนี้ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์
    </p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()