from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from pydantic import BaseModel, Field, field_validator
from typing import Literal

# Resolve path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "loan_model.pkl")

model = joblib.load(model_path)

app = FastAPI()

class LoanRequest(BaseModel):
    Applicant_Income: float = Field(..., ge=0.0, description="Applicant's monthly income")
    Coapplicant_Income: float = Field(..., ge=0.0, description="Co-applicant's monthly income")
    Employment_Status: Literal['Salaried', 'Self-employed', 'Contract', 'Unemployed']
    Age: int = Field(..., ge=18, le=120, description="Age of the applicant")
    Marital_Status: Literal['Married', 'Single']
    Dependents: int = Field(..., ge=0, le=20, description="Number of dependents")
    Credit_Score: int = Field(..., ge=300, le=850, description="Credit score of the applicant")
    Existing_Loans: int = Field(..., ge=0, le=50, description="Number of existing loans")
    DTI_Ratio: float = Field(..., ge=0.0, le=1.0, description="Debt-to-Income ratio (0.0 to 1.0)")
    Savings: float = Field(..., ge=0.0, description="Savings amount")
    Collateral_Value: float = Field(..., ge=0.0, description="Collateral value")
    Loan_Amount: float = Field(..., ge=0.0, description="Requested loan amount")
    Loan_Term: int = Field(..., ge=1, le=600, description="Loan term in months")
    Loan_Purpose: Literal['Personal', 'Car', 'Business', 'Home', 'Education']
    Property_Area: Literal['Urban', 'Semiurban', 'Rural']
    Education_Level: Literal['Not Graduate', 'Graduate']
    Gender: Literal['Female', 'Male']
    Employer_Category: Literal['Private', 'Government', 'Unemployed', 'MNC', 'Business']

    @field_validator(
        'Employment_Status', 'Marital_Status', 'Loan_Purpose',
        'Property_Area', 'Education_Level', 'Gender', 'Employer_Category',
        mode='before'
    )
    @classmethod
    def normalize_strings(cls, v):
        if isinstance(v, str):
            val = v.strip().lower()
            mapping = {
                "salaried": "Salaried",
                "self-employed": "Self-employed",
                "contract": "Contract",
                "unemployed": "Unemployed",
                "married": "Married",
                "single": "Single",
                "personal": "Personal",
                "car": "Car",
                "business": "Business",
                "home": "Home",
                "education": "Education",
                "urban": "Urban",
                "semiurban": "Semiurban",
                "rural": "Rural",
                "not graduate": "Not Graduate",
                "graduate": "Graduate",
                "female": "Female",
                "male": "Male",
                "private": "Private",
                "government": "Government",
                "mnc": "MNC",
            }
            return mapping.get(val, v)
        return v

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
    try:  
        prediction = model.predict(
            input_df
        )   
    except Exception as e:
        print("Something went wrong")
        print(e)
    
    try:
        probability = model.predict_proba(
            input_df
        )
    except Exception as e:
        print("Something went wrong")
        print(e)
        
        
    confidence = float(
        probability[0][prediction[0]]
    )
    
    label_map = {
    0: "Not Approved",
    1: "Approved"
    }
    
    loan_status = label_map[
    prediction[0]
    ]
    
    return {
        "loan_status": loan_status,
        "confidence": f"{confidence:.2%}" 
    }   




