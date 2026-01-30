from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import mlflow
import mlflow.xgboost
import pickle

app = FastAPI(title="Alzheimer Predictor API")
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# =========================
# MLflow config
# =========================
mlflow.set_tracking_uri(
    "file:///C:/Users/user/OneDrive/Documentos/SEXTO%20CICLO/EXAMENFINALIA/Notebook/mlruns"
)


run_id = "a9302cdf7df7439d8a59ea7c3fb148ff"

prep_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="preprocessor/preprocessor.pkl"
)

with open(prep_path, "rb") as f:
    transformador = pickle.load(f)

modelo = mlflow.xgboost.load_model("models:/Alzheimer_XGBoost/1")



# =========================
# FEATURE ENGINEERING
# =========================
def build_features(data: pd.DataFrame):

    data['age_mmse_interaction'] = data['Age'] * data['MMSE']
    data['cognitive_decline_score'] = data['MMSE'] + data['FunctionalAssessment'] + data['ADL']
    data['vascular_risk_score'] = data['Hypertension'] + data['CardiovascularDisease'] + data['Diabetes'] + data['Smoking']
    data['cholesterol_ratio'] = data['CholesterolLDL'] / (data['CholesterolHDL'] + 0.01)
    data['bp_ratio'] = data['SystolicBP'] / (data['DiastolicBP'] + 0.01)

    data['symptom_count'] = (
        data['Confusion'] +
        data['Disorientation'] +
        data['PersonalityChanges'] +
        data['DifficultyCompletingTasks'] +
        data['Forgetfulness'] +
        data['MemoryComplaints'] +
        data['BehavioralProblems']
    )

    data['lifestyle_score'] = data['PhysicalActivity'] + data['DietQuality'] + data['SleepQuality']
    data['age_group'] = pd.cut(data['Age'], bins=[59, 70, 80, 91], labels=[0, 1, 2])

    return data


# =========================
# HOME FRONTEND
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,

    Age: int = Form(...),
    Gender: int = Form(...),
    BMI: float = Form(...),
    MMSE: int = Form(...),
    FunctionalAssessment: int = Form(...),
    ADL: int = Form(...),

    CholesterolLDL: float = Form(...),
    CholesterolHDL: float = Form(...),
    SystolicBP: float = Form(...),
    DiastolicBP: float = Form(...),

    PhysicalActivity: int = Form(...),
    DietQuality: int = Form(...),
    SleepQuality: int = Form(...),

    # checkboxes
    Hypertension: str | None = Form(None),
    Diabetes: str | None = Form(None),
    Smoking: str | None = Form(None),
    CardiovascularDisease: str | None = Form(None),
    MemoryComplaints: str | None = Form(None),
    BehavioralProblems: str | None = Form(None),
    Confusion: str | None = Form(None),
    Disorientation: str | None = Form(None),
    PersonalityChanges: str | None = Form(None),
    DifficultyCompletingTasks: str | None = Form(None),
    Forgetfulness: str | None = Form(None),
):

    # convertir checkbox → 0/1
    def chk(x):
        if x is None:
            return 0
        x_str = str(x).lower()
        return 1 if x_str in ("on", "1", "true", "t", "yes") else 0

    data = pd.DataFrame([{
        "Age": Age,
        "Gender": Gender,
        "BMI": BMI,
        "MMSE": MMSE,
        "FunctionalAssessment": FunctionalAssessment,
        "ADL": ADL,

        "Hypertension": chk(Hypertension),
        "Diabetes": chk(Diabetes),
        "Smoking": chk(Smoking),
        "CardiovascularDisease": chk(CardiovascularDisease),

        "MemoryComplaints": chk(MemoryComplaints),
        "BehavioralProblems": chk(BehavioralProblems),
        "Confusion": chk(Confusion),
        "Disorientation": chk(Disorientation),
        "PersonalityChanges": chk(PersonalityChanges),
        "DifficultyCompletingTasks": chk(DifficultyCompletingTasks),
        "Forgetfulness": chk(Forgetfulness),

        "CholesterolLDL": CholesterolLDL,
        "CholesterolHDL": CholesterolHDL,
        "SystolicBP": SystolicBP,
        "DiastolicBP": DiastolicBP,

        "PhysicalActivity": PhysicalActivity,
        "DietQuality": DietQuality,
        "SleepQuality": SleepQuality,

        # defaults restantes
        "Ethnicity":0,"EducationLevel":0,"AlcoholConsumption":0,
        "FamilyHistoryAlzheimers":0,"Depression":0,"HeadInjury":0,
        "CholesterolTotal":0,"CholesterolTriglycerides":0,
    }])

    data = build_features(data)

    X = transformador.transform(data)

    pred = modelo.predict(X)[0]

    result = "🧠 ALTO RIESGO DE ALZHEIMER" if pred == 1 else "✅ BAJO RIESGO DE ALZHEIMER"

    return templates.TemplateResponse("index.html",
        {"request": request, "result": result}
    )
