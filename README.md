# 🧠 Alzheimer Risk Prediction System using Machine Learning

An intelligent risk prediction system that estimates the probability of Alzheimer’s disease using clinical, demographic, lifestyle, and cognitive assessment data.

The system is powered by XGBoost for machine learning and FastAPI for real-time inference through a web-based interface.

Disclaimer: This system provides risk estimation only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical evaluation.

## Key Features

✅ Supervised Machine Learning model using XGBoost

✅ Integration of clinical, lifestyle, and cognitive variables

✅ Advanced feature engineering pipeline

✅ Model tracking and versioning with MLflow

✅ RESTful API built with FastAPI

✅ Interactive web frontend (HTML + CSS)

✅ Real-time predictions

✅ Designed for retraining and continuous improvement

##  Machine Learning Model
### Algorithm
```
XGBoost Classifier
```

## Input Data Categories

* Demographic information

* Medical history

* Lifestyle factors

* Clinical measurements

* Cognitive and functional assessments

* Symptom indicators

## Engineered Features

* Cognitive decline score

* Vascular risk score

* Lifestyle score

* Symptom count

* Age interaction features

* Clinical ratios

## Methodology
The system follows a complete machine learning lifecycle, including initial training and continuous improvement through retraining:

1. Exploratory Data Analysis (EDA)
Statistical analysis and visualization to understand feature distributions, correlations, and potential biases.

2. Data Cleaning and Preprocessing
Handling missing values, scaling numerical variables, and encoding categorical features.

3. Feature Engineering
Creation of domain-driven features such as cognitive scores, lifestyle indices, clinical ratios, and interaction terms.

4. Model Training with XGBoost
Supervised learning using gradient boosting decision trees optimized for tabular medical data.

5. Model Evaluation and Validation
Performance evaluation using metrics such as accuracy, precision, recall, F1-score, and confusion matrix analysis.

6. Model Tracking and Versioning (MLflow)
Logging experiments, parameters, metrics, and artifacts to enable reproducibility and model comparison.

7. Model Retraining (Incremental Improvement)
The model supports retraining with newly collected patient data, allowing continuous performance improvement while preserving historical knowledge.

8. Deployment using FastAPI
The trained model is exposed as a REST API for real-time inference.

9. Web Interface Integration
A user-friendly web form allows interactive data input and instant risk prediction.

## Project Structure
```

Project/
│
│
├── fastapi-alzheimer/
|   ├── templates/
│   |  └── index.html 
│   ├── static/
│   |  └── style.css
│   ├── app.py  
│   ├── retrain.py
|   ├── review_and_label.py 
│
├── notebooks/
│   ├── alzheimer-disease-prediction-exploratory-analysis.ipynb
│   ├── transformation-and-processing-of-variables.ipynb
│   ├── training-with-xgboost.ipynb
│   ├── prediction-new-patients.ipynb
│   └── incremental_retraining.ipynb
│
└── README.md
```

## Technology Stack
### Machine Learning & Data Science

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* MLflow
* Backend
* FastAPI
* Uvicorn
  
### Frontend

* HTML
* CSS

### Visualization & Analysis

* Matplotlib
* Seaborn
* Plotly

## Running the Application
Start the API
```uvicorn app:app --reload```

### Open in browser
```http://127.0.0.1:8000```


## Disclaimer

This project is intended only for educational and research purposes.
It does not replace professional medical diagnosis or clinical decision-making.

## 👨‍💻 Authors

* Diego Alexander Bravo Valdiviezo

* Ariel Paltán 

 Computer Science Students
