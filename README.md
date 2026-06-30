# Loan Approval Prediction System

A machine learning project built to understand how trained models move beyond notebooks and become usable software systems.

Unlike tutorial-style ML projects that focus mainly on model accuracy, this project focuses on learning the **engineering workflow around machine learning systems** — from preprocessing pipelines and model training to backend integration, API validation, and deployment architecture.

> **Status: Active Development**


## Project Goal

The objective of this project is to understand how machine learning systems are built outside experimentation environments and integrated into real applications.

The focus is on learning engineering practices such as:

* Building reproducible preprocessing pipelines
* Preventing data leakage during training
* Model serialization and artifact management
* Backend API design for inference
* Schema validation and input normalization
* Understanding model serving architecture
* Designing systems with deployment constraints in mind

This project prioritizes **engineering workflow and system design**, not just model accuracy.



## Current Progress

### Completed

* Dataset cleaning and preprocessing
* Exploratory data analysis in Jupyter Notebook
* Model training and evaluation in notebook
* Hyperparameter tuning using GridSearchCV
* Model serialization using `joblib` (`loan_model.pkl`)
* FastAPI backend for model inference
* API endpoint for loan prediction
* Returning prediction confidence score
* Input validation using Pydantic
* Input normalization using custom validators

### In Progress

* Frontend integration with backend
* Building reproducible training pipeline (`train.py`)
* Better backend exception handling
* API testing and edge-case validation

### Planned

* Docker containerization
* Cloud deployment
* Structured logging
* Health check endpoint (`/health`)
* CI/CD pipeline
* Database integration for prediction history


## Dataset

Dataset used:

```text id="ds1"
loan_approval_data.csv
```

### Numerical Features

* Applicant_Income
* Coapplicant_Income
* Age
* Dependents
* Credit_Score
* Existing_Loans
* DTI_Ratio
* Savings
* Collateral_Value
* Loan_Amount
* Loan_Term

### Categorical Features

* Employment_Status
* Marital_Status
* Loan_Purpose
* Property_Area
* Education_Level
* Gender
* Employer_Category

### Target Variable

```text id="ds2"
Loan_Approved
```


## Machine Learning Pipeline

Current model:

```text id="ml1"
Logistic Regression
```

Workflow implemented:

```text id="ml2"
1. Load dataset
2. Remove rows with missing target values
3. Drop Applicant_ID
4. Train-Test Split
5. Automatic feature detection
6. Numerical preprocessing pipeline
7. Categorical preprocessing pipeline
8. ColumnTransformer integration
9. Model pipeline construction
10. Hyperparameter tuning
11. Serialize final trained model
```

### Train-Test Split

```python id="ml3"
test_size=0.2
random_state=42
stratify=y
```

### Numerical Pipeline

```python id="ml4"
SimpleImputer(strategy="mean")
StandardScaler()
```

### Categorical Pipeline

```python id="ml5"
SimpleImputer(strategy="most_frequent")
OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)
```

### Pipeline Architecture

```text id="ml6"
Preprocessing Pipeline
        ↓
LogisticRegression
```

Built using:

* Pipeline
* ColumnTransformer


## Hyperparameter Tuning

Used:

```text id="hp1"
GridSearchCV
```

Search space:

```python id="hp2"
{
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["lbfgs", "liblinear"]
}
```


## Model Serialization

The trained pipeline is saved using:

```text id="ser1"
joblib
```

Generated artifact:

```text id="ser2"
loan_model.pkl
```

Serialized model contains:

* Imputer state
* Scaler state
* Encoder categories
* Trained model weights

The full preprocessing pipeline and model are stored together to ensure consistent inference.


## Backend Architecture

Backend stack:

* FastAPI
* Uvicorn
* Pydantic

Current inference flow:

```text id="be1"
User Request
      ↓
JSON Payload
      ↓
FastAPI Route
      ↓
Pydantic Validation
      ↓
Convert to Dictionary
      ↓
Convert to Pandas DataFrame
      ↓
Pass into Loaded Model Pipeline
      ↓
model.predict()
      ↓
model.predict_proba()
      ↓
Return Response
```


## Model Loading Strategy

The model is loaded once during server startup.

```python id="be2"
model = joblib.load(...)
```

Flow:

```text id="be3"
Server Startup
      ↓
Load Model into RAM
      ↓
Reuse Loaded Model for All Requests
```

This avoids loading the model from disk on every incoming request.


## API Validation Layer

Validation implemented using:

```text id="val1"
Pydantic
```

### Numeric Validation

```python id="val2"
Age: int = Field(
    ge=18,
    le=120
)

Credit_Score: int = Field(
    ge=300,
    le=850
)

Dependents: int = Field(
    ge=0,
    le=20
)

Applicant_Income: float = Field(
    ge=0
)
```

### Categorical Validation

Implemented using:

```python id="val3"
Literal[]
```

Example:

```python id="val4"
Gender:
Literal["Male", "Female"]

Employment_Status:
Literal[
    "Salaried",
    "Self-employed",
    "Contract",
    "Unemployed"
]
```

Requests with invalid values are rejected before reaching the model.


## Input Normalization

Implemented custom validators to normalize safe user input before validation.

Examples:

```text id="norm1"
male   → Male
MALE   → Male
female → Female
```

Implemented using:

```python id="norm2"
@field_validator(..., mode="before")
```

Flow:

```text id="norm3"
Request Arrives
      ↓
Normalize Input
      ↓
Validate Allowed Values
      ↓
Execute Route
```


## API Response

Earlier response:

```json id="api1"
{
  "prediction": 1
}
```

Current response:

```json id="api2"
{
  "loan_status": "Approved",
  "confidence": "82.41%"
}
```

Confidence score is derived from:

```python id="api3"
prediction = model.predict()

probability = model.predict_proba()

confidence =
probability[0][prediction[0]]
```


## Repository Structure

```text id="repo1"
loan-prediction-system/
│
├── backend/
│   ├── loan_model.pkl
│   └── main.py
│
├── frontend/
│   ├── src/
│   └── package.json
│
├── training/
│   ├── loan_approval_data.csv
│   ├── LoanPredictior.ipynb
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```


## Running the Project

### Backend

```bash id="run1"
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

### Frontend

<!-- ```bash id="run2" -->
<!--  cd frontend -->
<!--  npm install -->
<!-- ``` -->

> Frontend integration is currently under development.


## Engineering Concepts Practiced

Through this project I am learning:

* Data leakage prevention
* Train/Test separation discipline
* sklearn pipeline design
* Model serialization and deserialization
* Backend API design for ML inference
* Runtime model loading behavior
* Request validation using schema contracts
* Input normalization before validation
* Confidence score interpretation
* Backend/frontend responsibility separation
* Deployment-oriented system design


## Upcoming Improvements

Planned next steps:

* Move notebook logic into reproducible `train.py`
* Improve backend exception handling
* Add API testing
* Implement structured logging
* Add health check endpoint
* Containerize using Docker
* Deploy backend and frontend


## Philosophy Behind This Project

The purpose of this project is not simply building a classifier.

It is an exercise in understanding how machine learning systems are engineered into usable applications.

```text id="end1"
Notebook
   ↓
Backend API
   ↓
Application Integration
   ↓
Deployment Architecture
   ↓
ML Engineering Mindset
```


## Author

**Aditya Tiwari**

B.Tech Student
Aspiring Machine Learning Engineer
