from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import pickle
import urllib.parse
import os
app = FastAPI(title="Alzheimer Predictor API")
from fastapi.staticfiles import StaticFiles
from retrain import handle_new_patient
from fastapi import Form
from retrain import save_to_buffer
from fastapi import FastAPI
from pathlib import Path
from retrain import retrain_incremental, MIN_PATIENTS

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

modelo = mlflow.xgboost.load_model("models:/Alzheimer_XGBoost/latest")


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


def validate_form(form: dict) -> list:
    """Validate essential form fields and ranges. Return list of error messages."""
    errors = []

    # Age
    age = form.get('Age')
    if age is None or not (5 <= int(age) <= 120):
        errors.append("Edad debe estar entre 40 y 120")

    # BMI
    bmi = form.get('BMI')
    try:
        bmi = float(bmi)
        if not (15 <= bmi <= 40):
            errors.append("BMI debe estar entre 15 y 40")
    except Exception:
        errors.append("BMI debe ser un número válido")

    # MMSE
    mmse = form.get('MMSE')
    if mmse is None or not (0 <= int(mmse) <= 30):
        errors.append("MMSE debe estar entre 0 y 30")

    # FunctionalAssessment / ADL
    fa = form.get('FunctionalAssessment')
    adl = form.get('ADL')
    if fa is None or not (0 <= int(fa) <= 10):
        errors.append("Evaluación Funcional debe estar entre 0 y 10")
    if adl is None or not (0 <= int(adl) <= 10):
        errors.append("ADL debe estar entre 0 y 10")

    # Blood pressure
    try:
        sys = float(form.get('SystolicBP', 0))
        dia = float(form.get('DiastolicBP', 0))
        if not (90 <= sys <= 180):
            errors.append("Presión sistólica fuera de rango (90-180)")
        if not (60 <= dia <= 120):
            errors.append("Presión diastólica fuera de rango (60-120)")
    except Exception:
        errors.append("Presión arterial debe ser numérica")

    # Cognitive/ratings
    try:
        pa = int(form.get('PhysicalActivity', 0))
        diet = int(form.get('DietQuality', 0))
        sleep = int(form.get('SleepQuality', 0))
        for name, val in (('Actividad Física', pa), ('Dieta', diet), ('Sueño', sleep)):
            if not (0 <= val <= 10):
                errors.append(f"{name} debe estar entre 0 y 10")
    except Exception:
        errors.append("Actividad/Dieta/Sueño debe ser numérico (0-10)")

    return errors


def save_prediction_pending(X_prep, pred, proba):
    os.makedirs("buffer", exist_ok=True)
    path = "buffer/predictions_pending.pkl"

    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        data = []

    data.append({
        "X": X_prep.tolist(),
        "predicted_label": int(pred),
        "confidence": float(proba)
    })

    with open(path, "wb") as f:
        pickle.dump(data, f)

def get_pending_and_buffer_info():
    # pendientes
    pending_path = Path("buffer/predictions_pending.pkl")
    if pending_path.exists():
        with open(pending_path, "rb") as f:
            pending = pickle.load(f)
    else:
        pending = []

    # buffer confirmados
    buffer_path = Path("buffer/new_patients.pkl")
    buffer_count = 0
    if buffer_path.exists():
        with open(buffer_path, "rb") as f:
            buffer = pickle.load(f)
            buffer_count = len(buffer["X"])

    can_retrain = buffer_count >= MIN_PATIENTS

    return pending, buffer_count, can_retrain


# =========================
# HOME FRONTEND
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    pending, buffer_count, can_retrain = get_pending_and_buffer_info()

    defaults = {
        "Age": 72,
        "Gender": 0,
        "BMI": 23.5,
        "MMSE": 28,
        "FunctionalAssessment": 5,
        "ADL": 1,
        "Hypertension": 1,
        "Diabetes": 0,
        "Smoking": 0,
        "CardiovascularDisease": 0,
        "MemoryComplaints": 0,
        "BehavioralProblems": 0,
        "Confusion": 0,
        "Disorientation": 0,
        "PersonalityChanges": 0,
        "DifficultyCompletingTasks": 1,
        "Forgetfulness": 0,
        "CholesterolLDL": 130,
        "CholesterolHDL": 50,
        "SystolicBP": 130,
        "DiastolicBP": 80,
        "PhysicalActivity": 5,
        "DietQuality": 2,
        "SleepQuality": 6,
        "Ethnicity": 1,
        "EducationLevel": 3,
        "AlcoholConsumption": 10,
        "FamilyHistoryAlzheimers": 0,
        "Depression": 0,
        "HeadInjury": 0,
        "CholesterolTotal": 200,
        "CholesterolTriglycerides": 150
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "form_data": defaults,
            "pending": pending,
            "buffer_count": buffer_count,
            "can_retrain": can_retrain,
            "min_patients": MIN_PATIENTS
        }
    )



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

    # additional fields from form
    Ethnicity: int = Form(...),
    EducationLevel: int = Form(...),
    AlcoholConsumption: float = Form(...),
    CholesterolTotal: float = Form(...),
    CholesterolTriglycerides: float = Form(...),

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
    FamilyHistoryAlzheimers: str | None = Form(None),
    Depression: str | None = Form(None),
    HeadInjury: str | None = Form(None),
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

        # campos restantes (mapear desde el formulario)
        "Ethnicity": Ethnicity,
        "EducationLevel": EducationLevel,
        "AlcoholConsumption": AlcoholConsumption,
        "FamilyHistoryAlzheimers": chk(FamilyHistoryAlzheimers),
        "Depression": chk(Depression),
        "HeadInjury": chk(HeadInjury),
        "CholesterolTotal": CholesterolTotal,
        "CholesterolTriglycerides": CholesterolTriglycerides,
    }])

    data = build_features(data)

    # Prepare form data dict for validation and template
    form_data = data.to_dict(orient="records")[0]

    # Server-side validation: return early with message if invalid
    errors = validate_form(form_data)
    if errors:
        msg = "; ".join(errors)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "msg": msg,
                "form_data": form_data
            }
        )

    X = transformador.transform(data)
    X_list = X.tolist()[0]

    proba = modelo.predict_proba(X)[0][1]  # probabilidad de Alzheimer
    pred = 1 if proba >= 0.5 else 0

    if proba < 0.30:
        result = "✅ Riesgo bajo de Alzheimer"
        advice = "No se detectan indicadores significativos. Se recomienda control periódico."
    elif proba < 0.60:
        result = "🟡 Riesgo moderado de Alzheimer"
        advice = "Se sugiere seguimiento clínico y evaluación cognitiva."
    else:
        result = "🔴 Riesgo alto de Alzheimer"
        advice = "Se recomienda evaluación médica especializada."

    confidence = round(proba * 100, 2)
    save_prediction_pending(X, pred, proba)


    pending, buffer_count, can_retrain = get_pending_and_buffer_info()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "advice": advice,
            "confidence": confidence,
            "form_data": form_data,
            "pending": pending,
            "buffer_count": buffer_count,
            "can_retrain": can_retrain,
            "min_patients": MIN_PATIENTS
        }
    )





@app.post("/confirm")
def confirm_patient(
    index: int = Form(...),
    true_label: int = Form(...)
):
    path = Path("buffer/predictions_pending.pkl")

    with open(path, "rb") as f:
        pending = pickle.load(f)

    # Caso inválido o ya procesado
    if index < 0 or index >= len(pending):
        pending, buffer_count, can_retrain = get_pending_and_buffer_info()
        return {
            "msg": "Caso ya procesado",
            "pending_count": len(pending),
            "buffer_count": buffer_count,
            "can_retrain": can_retrain,
            "min_patients": MIN_PATIENTS
        }

    # Extraer caso
    item = pending.pop(index)
    predicted_label = item["predicted_label"]

    matches = predicted_label == int(true_label)
    match_msg = "Coincide con la predicción" if matches else "Diferente a la predicción"

    # Guardar en buffer
    save_to_buffer(
        X_prep=np.array(item["X"]),
        y=[int(true_label)]
    )

    # Guardar pendientes restantes
    if pending:
        with open(path, "wb") as f:
            pickle.dump(pending, f)
    else:
        path.unlink(missing_ok=True)

    # Estado ACTUALIZADO
    pending, buffer_count, can_retrain = get_pending_and_buffer_info()

    response_msg = f"Paciente guardado ({match_msg})"
    if can_retrain:
        response_msg += f". ¡Listo para reentrenar! ({buffer_count} pacientes confirmados)"

    return {
        "msg": response_msg,
        "pending_count": len(pending),
        "buffer_count": buffer_count,
        "can_retrain": can_retrain,
        "min_patients": MIN_PATIENTS
    }


@app.get("/pending")
def view_pending():
    path = Path("buffer/predictions_pending.pkl")
    if not path.exists():
        return {"pending": []}

    with open(path, "rb") as f:
        pending = pickle.load(f)

    return {"pending": pending}


@app.post("/retrain")
def retrain_model():
    buffer_path = Path("buffer/new_patients.pkl")

    if not buffer_path.exists():
        return {
            "msg": "No hay pacientes confirmados",
            "buffer_count": 0,
            "can_retrain": False
        }

    with open(buffer_path, "rb") as f:
        buffer = pickle.load(f)

    if len(buffer["X"]) < MIN_PATIENTS:
        return {
            "msg": f"Se necesitan al menos {MIN_PATIENTS} pacientes",
            "buffer_count": len(buffer["X"]),
            "can_retrain": False
        }

    msg = retrain_incremental()

    return {
        "msg": msg,
        "buffer_count": 0,
        "can_retrain": False
    }
