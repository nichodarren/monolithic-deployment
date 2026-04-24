import streamlit as st
import joblib
import numpy as np
import pandas as pd

from train_placementPipeline import add_feature_engineering

# Load the machine learning models
classifier = joblib.load('placement_classifier.pkl')
regressor  = joblib.load('salary_regressor.pkl')

def main():
    st.title('Student Placement & Salary Prediction')

    gender = st.selectbox("Gender", ["Male", "Female"])
    branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "CE", "ME"])
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)
    tenth_percentage = st.slider("10th Percentage", 0.0, 100.0, 70.0)
    twelfth_percentage = st.slider("12th Percentage", 0.0, 100.0, 70.0)
    backlogs = st.number_input("Backlogs", 0, 20, 0)

    study_hours_per_day = st.slider("Study Hours per Day", 0.0, 12.0, 4.0)
    attendance_percentage = st.slider("Attendance (%)", 0.0, 100.0, 75.0)

    projects_completed = st.number_input("Projects Completed", 0, 50, 3)
    internships_completed = st.number_input("Internships Completed", 0, 10, 1)
    hackathons_participated = st.number_input("Hackathons Participated", 0, 20, 2)
    certifications_count = st.number_input("Certifications Count", 0, 30, 2)

    coding_skill_rating = st.slider("Coding Skill Rating", 1, 5, 3)
    communication_skill_rating = st.slider("Communication Skill Rating", 1, 5, 3)
    aptitude_skill_rating = st.slider("Aptitude Skill Rating", 1, 5, 3)

    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    stress_level = st.slider("Stress Level", 0, 10, 5)

    part_time_job = st.selectbox("Part Time Job", ["Yes", "No"])
    family_income_level = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    extracurricular_involvement = st.selectbox("Extracurricular Involvement", ["Low", "Medium", "High"])

    if st.button('Make Prediction'):

        features = {
            "gender": gender,
            "branch": branch,
            "cgpa": cgpa,
            "tenth_percentage": tenth_percentage,
            "twelfth_percentage": twelfth_percentage,
            "backlogs": backlogs,
            "study_hours_per_day": study_hours_per_day,
            "attendance_percentage": attendance_percentage,
            "projects_completed": projects_completed,
            "internships_completed": internships_completed,
            "coding_skill_rating": coding_skill_rating,
            "communication_skill_rating": communication_skill_rating,
            "aptitude_skill_rating": aptitude_skill_rating,
            "hackathons_participated": hackathons_participated,
            "certifications_count": certifications_count,
            "sleep_hours": sleep_hours,
            "stress_level": stress_level,
            "part_time_job": part_time_job,
            "family_income_level": family_income_level,
            "city_tier": city_tier,
            "internet_access": internet_access,
            "extracurricular_involvement": extracurricular_involvement,
        }

        df = pd.DataFrame([features])
        df = add_feature_engineering(df)

        placement_result = classifier.predict(df)[0]
        st.success(f"Placement Prediction: {placement_result}")

        if str(placement_result) == "Placed":
            salary_result = regressor.predict(df)[0]
            st.success(f"Predicted Salary (LPA): {salary_result:.2f}")
        else:
            st.info("Salary prediction skipped (student not predicted as Placed).")


if __name__ == '__main__':
    main()
