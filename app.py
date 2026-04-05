
import gradio as gr
import pickle
import pandas as pd
import numpy as np

# Load trained pipeline
with open("modelRF.pkl", "rb") as file:
    pipeline = pickle.load(file)

FEATURES = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently', 
    'BusinessTravel_Travel_Rarely', 'Department_Research & Development', 'Department_Sales', 
    'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical', 
    'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Male', 
    'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager', 
    'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist', 
    'JobRole_Sales Executive', 'JobRole_Sales Representative', 'MaritalStatus_Married', 
    'MaritalStatus_Single', 'OverTime_Yes', 'IncomePerYear', 'YearsAtCompanyRatio', 'DistanceIncome'
]

def predict_attrition(*args):

    input_values = list(args)
    
    data = {
        "Age": float(input_values[0]),
        "DailyRate": float(input_values[1]),
        "DistanceFromHome": float(input_values[2]),
        "Education": float(input_values[3]),
        "EnvironmentSatisfaction": float(input_values[4]),
        "HourlyRate": float(input_values[5]),
        "JobInvolvement": float(input_values[6]),
        "JobLevel": float(input_values[7]),
        "JobSatisfaction": float(input_values[8]),
        "MonthlyIncome": float(input_values[9]),
        "MonthlyRate": float(input_values[10]),
        "NumCompaniesWorked": float(input_values[11]),
        "PercentSalaryHike": float(input_values[12]),
        "PerformanceRating": float(input_values[13]),
        "RelationshipSatisfaction": float(input_values[14]),
        "StockOptionLevel": float(input_values[15]),
        "TotalWorkingYears": float(input_values[16]),
        "TrainingTimesLastYear": float(input_values[17]),
        "WorkLifeBalance": float(input_values[18]),
        "YearsAtCompany": float(input_values[19]),
        "YearsInCurrentRole": float(input_values[20]),
        "YearsSinceLastPromotion": float(input_values[21]),
        "YearsWithCurrManager": float(input_values[22]),
        "IncomePerYear": float(input_values[23]),
        "YearsAtCompanyRatio": float(input_values[24]),
        "DistanceIncome": float(input_values[25]),
        
        # Categorical Logic (Dropdowns are at the end of the input list)
        "BusinessTravel_Travel_Frequently": input_values[26] == "Frequently",
        "BusinessTravel_Travel_Rarely": input_values[26] == "Rarely",
        "Department_Research & Development": input_values[27] == "R&D",
        "Department_Sales": input_values[27] == "Sales",
        "EducationField_Life Sciences": input_values[28] == "Life Sciences",
        "EducationField_Marketing": input_values[28] == "Marketing",
        "EducationField_Medical": input_values[28] == "Medical",
        "EducationField_Other": input_values[28] == "Other",
        "EducationField_Technical Degree": input_values[28] == "Technical Degree",
        "Gender_Male": input_values[29] == "Male",
        "JobRole_Human Resources": input_values[30] == "HR",
        "JobRole_Laboratory Technician": input_values[30] == "Lab Tech",
        "JobRole_Manager": input_values[30] == "Manager",
        "JobRole_Manufacturing Director": input_values[30] == "Mfg Director",
        "JobRole_Research Director": input_values[30] == "Res Director",
        "JobRole_Research Scientist": input_values[30] == "Res Scientist",
        "JobRole_Sales Executive": input_values[30] == "Sales Exec",
        "JobRole_Sales Representative": input_values[30] == "Sales Rep",
        "MaritalStatus_Married": input_values[31] == "Married",
        "MaritalStatus_Single": input_values[31] == "Single",
        "OverTime_Yes": input_values[32] == "Yes"
    }

    #  column order
    input_df = pd.DataFrame([data])[FEATURES]
    
    # prediction 
    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0]
    
    
    confidence = f"{round(np.max(prob)*100, 2)}%"
    
    return f"Attrition Probability: {confidence}"

# UI Inputs 
inputs = [
    gr.Number(label="Age"), gr.Number(label="Daily Rate"), 
    gr.Number(label="Distance From Home"),
    gr.Number(label="Education"),
    gr.Slider(1,4, label="Env Satisfaction"),
    gr.Number(label="Hourly Rate"),
    gr.Slider(1,4, label="Job Involvement"), 
    gr.Number(label="Job Level"), 
    gr.Slider(1,4, label="Job Satisfaction"),
    gr.Number(label="Monthly Income"), 
    gr.Number(label="Monthly Rate"), 
    gr.Number(label="Num Companies Worked"),
    gr.Number(label="Percent Salary Hike"), 
    gr.Number(label="Performance Rating"), 
    gr.Slider(1,4, label="Relationship Sat"),
    gr.Number(label="Stock Option Level"), 
    gr.Number(label="Total Working Years"), 
    gr.Number(label="Training Times"),
    gr.Number(label="Work Life Balance"), 
    gr.Number(label="Years At Company"), gr.Number(label="Years In Role"),
    gr.Number(label="Years Since Promotion"), gr.Number(label="Years With Manager"),
    gr.Number(label="Income Per Year"), gr.Number(label="Years At Co Ratio"), gr.Number(label="Distance Income"),
    
    # Categoricals
    gr.Dropdown(["Rarely", "Frequently", "Non-Travel"], label="Travel"),
    gr.Dropdown(["R&D", "Sales", "HR"], label="Department"),
    gr.Dropdown(["Life Sciences", "Marketing", "Medical", "Technical Degree", "Other"], label="Field"),
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Dropdown(["Sales Exec", "Res Scientist", "Lab Tech", "Mfg Director", "Manager", "Sales Rep", "Res Director", "HR"], label="Role"),
    gr.Radio(["Single", "Married", "Divorced"], label="Marital Status"),
    gr.Radio(["Yes", "No"], label="OverTime")
]

demo = gr.Interface(fn=predict_attrition, inputs=inputs, outputs="text")
demo.launch(debug=True)







# import gradio as gr
# import pickle
# import pandas as pd

# Load trained pipeline
# with open("modelRF.pkl", "rb") as file:
#     pipeline = pickle.load(file)

# # Prediction function
# def predict_attrition(
    
#     Age,
#     Education,
#     MonthlyRate,
#     DailyRate,
#     JobLevel,
#     monthlyIncome,
#     DistanceFromHome,
#     YearsAtCompany,
#     TotalWorkingYears,
#     YearsInCurrentRole,
#     YearsSinceLastPromotion,
#     YearsWithCurrManager,
#     TrainingTimesLastYear,
#     NumCompaniesWorked,
#     JobInvolvement,
#     JobSatisfaction,
#     EnvironmentSatisfaction,
#     RelationshipSatisfaction,
#     WorkLifeBalance,
#     StockOptionLevel,
#     PerformanceRating,
#     PercentSalaryHike,
#     HourlyRate,
#     DistanceIncome,
#     IncomePerYear,
#     YearsAtCompanyRatio,
#     OverTime,
    
    
# ):
#     OverTime_binary = 1 if OverTime == "Yes" else 0

#     input_df = pd.DataFrame([[
#     Age,
#     DailyRate,
#     MonthlyRate,
#     JobLevel,
#     monthlyIncome,
#     Education,
#     DistanceFromHome,
#     YearsAtCompany,
#     TotalWorkingYears,
#     YearsInCurrentRole,
#     YearsSinceLastPromotion,
#     YearsWithCurrManager,
#     TrainingTimesLastYear,
#     NumCompaniesWorked,
#     JobInvolvement,
#     JobSatisfaction,
#     EnvironmentSatisfaction,
#     RelationshipSatisfaction,
#     WorkLifeBalance,
#     StockOptionLevel,
#     PerformanceRating,
#     PercentSalaryHike,
#     HourlyRate,
#     DistanceIncome,
#     IncomePerYear,
#     YearsAtCompanyRatio,
#     OverTime_binary
# ]],
# columns=[
#     "Age",
#     "DailyRate",
#     "MonthlyRate",
#     "JobLevel",
#     "MonthlyIncome",
#     "Education",
#     "DistanceFromHome",
#     "YearsAtCompany",
#     "TotalWorkingYears",
#     "YearsInCurrentRole",
#     "YearsSinceLastPromotion",
#     "YearsWithCurrManager",
#     "TrainingTimesLastYear",
#     "NumCompaniesWorked",
#     "JobInvolvement",
#     "JobSatisfaction",
#     "EnvironmentSatisfaction",
#     "RelationshipSatisfaction",
#     "WorkLifeBalance",
#     "StockOptionLevel",
#     "PerformanceRating",
#     "PercentSalaryHike",
#     "HourlyRate",
#     "DistanceIncome",
#     "IncomePerYear",
#     "YearsAtCompanyRatio",
#     "OverTime"
# ])

#     prediction = pipeline.predict(input_df)[0]

#     return "Yes (Will Leave)" if prediction == 1 else "No (Will Stay)"


# # UI Inputs
# inputs = [
#     gr.Number(label="Age"),
#     gr.Number(label="Monthly Income"),
#     gr.Number(label="Distance From Home"),
#     gr.Number(label="Years At Company"),
#     gr.Number(label="Total Working Years"),
#     gr.Number(label="Years In Current Role"),
#     gr.Number(label="Years Since Last Promotion"),
#     gr.Number(label="Years With Current Manager"),
#     gr.Number(label="Training Times Last Year"),
#     gr.Number(label="Number of Companies Worked"),
#     gr.Number(label="Job Involvement"),
#     gr.Number(label="Job Satisfaction"),
#     gr.Number(label="Environment Satisfaction"),
#     gr.Number(label="Relationship Satisfaction"),
#     gr.Number(label="Work Life Balance"),
#     gr.Number(label="Stock Option Level"),
#     gr.Number(label="Performance Rating"),
#     gr.Number(label="Percent Salary Hike"),
#     gr.Number(label="Hourly Rate"),
#     gr.Number(label="Distance Income"),
#     gr.Number(label="Income Per Year"),
#     gr.Number(label="Years At Company Ratio"),
#     gr.Radio(label="OverTime", choices=["Yes", "No"]),
#     gr.Number(label="Daily Rate"),
#     gr.Number(label="Monthly Rate"),
#     gr.Number(label="Job Level"),
#     gr.Number(label="Education"),


# ]

# # Gradio App
# app = gr.Interface(
#     fn=predict_attrition,
#     inputs=inputs,
#     outputs="text",
#     title="Employee Attrition Prediction",
#     description="Enter employee details to predict whether they will leave or stay."
# )

# app.launch()