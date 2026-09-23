import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("attrition_model.pkl")
preprocessor = joblib.load("attrition_preprocessor.pkl")

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊"
)

st.title("📊 Employee Attrition Prediction")
st.write("Enter employee details to predict the risk of attrition.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 60, 30)

    business_travel = st.selectbox(
        "Business Travel",
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )

    daily_rate = st.number_input("Daily Rate", 100, 1500, 800)

    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )

    distance_from_home = st.number_input(
        "Distance From Home", 1, 30, 5
    )

    education = st.number_input(
        "Education", 1, 5, 3
    )

    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )

    environment_satisfaction = st.number_input(
        "Environment Satisfaction", 1, 4, 3
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    hourly_rate = st.number_input(
        "Hourly Rate", 30, 100, 60
    )

    job_involvement = st.number_input(
        "Job Involvement", 1, 4, 3
    )

    job_level = st.number_input(
        "Job Level", 1, 5, 2
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    job_satisfaction = st.number_input(
        "Job Satisfaction", 1, 4, 3
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )


with col2:

    monthly_income = st.number_input(
        "Monthly Income", 1000, 20000, 4000
    )

    monthly_rate = st.number_input(
        "Monthly Rate", 2000, 27000, 15000
    )

    num_companies_worked = st.number_input(
        "Number of Companies Worked", 0, 10, 2
    )

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )

    percent_salary_hike = st.number_input(
        "Percent Salary Hike", 10, 25, 15
    )

    performance_rating = st.number_input(
        "Performance Rating", 1, 4, 3
    )

    relationship_satisfaction = st.number_input(
        "Relationship Satisfaction", 1, 4, 3
    )

    stock_option_level = st.number_input(
        "Stock Option Level", 0, 3, 0
    )

    total_working_years = st.number_input(
        "Total Working Years", 0, 40, 6
    )

    training_times_last_year = st.number_input(
        "Training Times Last Year", 0, 10, 3
    )

    work_life_balance = st.number_input(
        "Work Life Balance", 1, 4, 3
    )

    years_at_company = st.number_input(
        "Years At Company", 0, 40, 3
    )

    years_in_current_role = st.number_input(
        "Years In Current Role", 0, 20, 2
    )

    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion", 0, 15, 1
    )

    years_with_curr_manager = st.number_input(
        "Years With Current Manager", 0, 20, 2
    )


if st.button("Predict Attrition"):

    employee_data = {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager
    }

    employee_df = pd.DataFrame([employee_data])

    employee_processed = preprocessor.transform(employee_df)

    probability = model.predict_proba(employee_processed)[0, 1]

    if probability >= 0.20:
        prediction = "Employee may leave"
    else:
        prediction = "Employee may stay"

    st.subheader("Prediction Result")

    st.write(
        f"Attrition Probability: **{probability * 100:.2f}%**"
    )

    if probability >= 0.20:
        st.warning(f"⚠️ {prediction}")
    else:
        st.success(f"✅ {prediction}")