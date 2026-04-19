


import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from config.config import ARTIFACT_PIPELINE_CLASS, ARTIFACT_PIPELINE_REG
from src.utils.io import load_artifact
from src.data.loader import feature_engineering

@st.cache_resource
def load_pipelines():
    return load_artifact(ARTIFACT_PIPELINE_CLASS), load_artifact(ARTIFACT_PIPELINE_REG)

model_placement, model_salary = load_pipelines()

def main():
    st.title("Student Placement and Salary Prediction")

    st.sidebar.header("Information")
    st.sidebar.write("This system used to predict job placement probabilities and salary estimates for students.")
    st.sidebar.write("Models used:")
    st.sidebar.write("- Classification : LightGBM")
    st.sidebar.write("- Regression : Random Forest with Optuna")

    with st.form("input_form"):
        st.subheader("Academic performance, skills, and experience")
        
        col1, col2 = st.columns(2)
        
        with col1:        
            student_id = st.text_input("Student ID",value="1")
            gender = st.selectbox("Gender", ["Male","Female"])
            ssc_percentage = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, value=80.0)
            hsc_percentage = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, value=80.0)
            degree_percentage = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=75.0)
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5)
            entrance_exam = st.number_input("Entrance Exam Score", min_value=0.0, max_value=100.0, value=70.0)
            attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=90.0)

        with col2:
            tech_skill = st.number_input("Technical Skill Score", min_value=0, max_value=100, value=80)
            soft_skill = st.number_input("Soft Skill Score", min_value=0, max_value=100, value=80)
            internship_count = st.number_input("Internship Count", min_value=0, value=0)
            live_projects = st.number_input("Live Projects Count", min_value=0,value=0)
            work_experience = st.number_input("Work Experience (Months)", min_value=0, value=0)
            certifications = st.number_input("Certifications Count", min_value=0, value=0)
            backlogs = st.number_input("Backlogs Count", min_value=0, value=0)
            extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

        submit = st.form_submit_button("Student Analysis")

    if submit:
        features = pd.DataFrame([{
            'student_id': student_id,
            'gender': gender,
            'ssc_percentage': ssc_percentage,
            'hsc_percentage': hsc_percentage,
            'degree_percentage': degree_percentage,
            'cgpa': cgpa,
            'entrance_exam_score': entrance_exam,
            'technical_skill_score': tech_skill,
            'soft_skill_score': soft_skill,
            'internship_count': internship_count,
            'live_projects': live_projects,
            'work_experience_months': work_experience,
            'certifications': certifications,
            'attendance_percentage': attendance,
            'backlogs': backlogs,
            'extracurricular_activities': extracurricular
        }])

       
        df_processed = feature_engineering(features)
        
       
        is_placed = model_placement.predict(df_processed)[0]
        
        st.divider()

        # Radar chart ui/ux
        st.subheader("Competence Visualization")
        
        
        backlog_stability = max(0, (10 - backlogs)*10) 
        cgpa_normalized = cgpa*10 
        
        categories = ['CGPA', 'Technical Skill', 'Soft Skill', 'Attendance', 'Backlog Stability', 'Live Projects']
        values = [cgpa_normalized, tech_skill, soft_skill, attendance, backlog_stability, live_projects*20]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Student Profile',
            line_color='green'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False
        )
        st.plotly_chart(fig)

        # Hasil Akhir
        st.subheader(f"Analysis Results for Student with ID : {student_id}")
        if is_placed == 1:
            st.success("Prediction Result: Placed")
            salary = model_salary.predict(df_processed)[0]
            st.info(f"Salary Package Estimation: {salary:.2f} LPA")
        else:
            st.error("Prediction Result: Not Placed")

if __name__ == "__main__":
    main()