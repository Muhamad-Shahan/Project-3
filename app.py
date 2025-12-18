import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Retention Radar",
    page_icon="👔",
    layout="centered"
)

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('retention_model.pkl')
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('model_columns.pkl')
        return model, scaler, columns
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

model, scaler, model_columns = load_assets()

# --- 3. UI: HEADER ---
st.title("👔 Retention Radar")
st.markdown("### HR Flight Risk Simulator")
st.write("Enter employee details below to predict the probability of turnover.")

# --- 4. INPUT FORM (SIDEBAR) ---
st.sidebar.header("Employee Profile")

# Satisfaction & Performance
satisfaction = st.sidebar.slider("Satisfaction Level", 0.0, 1.0, 0.50, help="0.0 = Unhappy, 1.0 = Very Happy")
evaluation = st.sidebar.slider("Last Evaluation", 0.0, 1.0, 0.70)
projects = st.sidebar.number_input("Number of Projects", 2, 7, 4)

# Workload & Tenure
hours = st.sidebar.slider("Average Monthly Hours", 96, 310, 200)
tenure = st.sidebar.number_input("Time at Company (Years)", 2, 10, 3)
accident = st.sidebar.checkbox("Work Accident?")
promotion = st.sidebar.checkbox("Promoted in last 5 years?")

# Department & Salary
dept = st.sidebar.selectbox("Department", [
    'sales', 'accounting', 'hr', 'technical', 'support', 
    'management', 'IT', 'product_mng', 'marketing', 'RandD'
])
salary = st.sidebar.select_slider("Salary Level", options=['low', 'medium', 'high'])

# --- 5. PREDICTION LOGIC ---
if st.sidebar.button("Run Prediction", type="primary"):
    if model is None:
        st.error("Model not loaded. Run setup_model.py first.")
    else:
        # 1. Prepare Input Dictionary
        input_data = {
            'satisfaction_level': satisfaction,
            'last_evaluation': evaluation,
            'number_project': projects,
            'average_montly_hours': hours,
            'time_spend_company': tenure,
            'Work_accident': 1 if accident else 0,
            'promotion_last_5years': 1 if promotion else 0,
            'salary_num': {'low': 1, 'medium': 2, 'high': 3}[salary]
        }
        
        # 2. Handle Department One-Hot Encoding (The Tricky Part!)
        # We start with 0 for all depts, then set the chosen one to 1
        for col in model_columns:
            if 'dept_' in col:
                input_data[col] = 0  # Initialize all depts to 0
        
        # If the chosen dept exists in our columns (e.g. dept_sales), set it to 1
        # Note: 'RandD' was the reference category if it was dropped, so we check carefully
        dept_col_name = f"dept_{dept}"
        if dept_col_name in input_data:
            input_data[dept_col_name] = 1
            
        # 3. Create DataFrame in the exact order the model expects
        input_df = pd.DataFrame([input_data])
        # Ensure columns are ordered correctly (reindex)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # 4. Scale Data
        input_scaled = scaler.transform(input_df)
        
        # 5. Predict
        # model.predict_proba returns [[prob_stay, prob_leave]]
        probability_leave = model.predict_proba(input_scaled)[0][1]
        
        # --- 6. VISUALIZATION ---
        st.markdown("---")
        
        # Layout Columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability_leave * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Flight Risk Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF4B4B" if probability_leave > 0.5 else "#2ECC71"},
                    'steps': [
                        {'range': [0, 50], 'color': "#f0f2f6"},
                        {'range': [50, 100], 'color': "#ffecec"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Text Insight
            st.markdown("### Risk Analysis")
            if probability_leave > 0.5:
                st.error("⚠️ **High Risk**")
                st.write("This employee is highly likely to leave.")
                
                # Simple Heuristics for recommendations
                if satisfaction < 0.5:
                    st.write("👉 **Key Factor:** Low Satisfaction.")
                if hours > 250:
                    st.write("👉 **Key Factor:** Overworked.")
            else:
                st.success("✅ **Safe**")
                st.write("This employee is likely to stay.")

else:
    st.info("👈 Adjust settings in the sidebar and click 'Run Prediction'")
