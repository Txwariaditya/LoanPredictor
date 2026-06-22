from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Resolve path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "loan_model.pkl")

model = joblib.load(model_path)

app = FastAPI()

class LoanRequest(BaseModel):    
    Applicant_Income : float
    Coapplicant_Income : float
    Employment_Status : str
    Age : int
    Marital_Status : str
    Dependents : int
    Credit_Score : int
    Existing_Loans : int
    DTI_Ratio : float
    Savings : float
    Collateral_Value : float
    Loan_Amount : float
    Loan_Term : int
    Loan_Purpose : str
    Property_Area : str
    Education_Level : str
    Gender : str
    Employer_Category : str



@app.get("/")
def home():
    return {
        "message": "Loan prediction API running"
    }
    
@app.post("/predict")
def predict(data: LoanRequest):

    input_dict = data.model_dump()

    input_df = pd.DataFrame(
        [input_dict]
    )

    prediction = model.predict(
        input_df
    )

    return {
        "prediction": int(prediction[0])
    }   