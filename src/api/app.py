import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from src.api.deps import get_predictor, model_is_ready
from src.api.schemas import (
    BatchPredictItem,
    BatchPredictRequest,
    BatchPredictResponse,
    EmployeeInput,
    HealthResponse,
    PredictResponse,
)
from src.inference import load_default_employee, risk_label

app = FastAPI(
    title='HR Attrition API',
    description='Прогноз вероятности увольнения сотрудника (IBM HR Attrition, RandomForest tuned)',
    version='1.0.0',
)


def _ensure_model():
    if not model_is_ready():
        raise HTTPException(
            status_code=503,
            detail='Модель не найдена',
        )


def _predict_frame(frame):
    predictor = get_predictor()
    probs = predictor.predict_proba(frame)
    preds = predictor.predict(frame)
    model_name = predictor.meta.get('model_name', 'RandomForest_tuned')
    out = []
    for i, prob in enumerate(probs):
        prob_f = float(prob)
        out.append(
            {
                'attrition_probability': round(prob_f, 4),
                'attrition_predicted': int(preds[i]),
                'risk_level': risk_label(prob_f),
                'model': model_name,
            }
        )
    return out


@app.get('/', include_in_schema=False)
def root():
    return RedirectResponse(url='/docs')


@app.get('/health', response_model=HealthResponse)
def health():
    ready = model_is_ready()
    name = None
    if ready:
        try:
            name = get_predictor().meta.get('model_name', 'RandomForest_tuned')
        except FileNotFoundError:
            ready = False
    return HealthResponse(status='ok', model_loaded=ready, model_name=name)


@app.get('/defaults')
def defaults():
    return load_default_employee()


@app.post('/predict', response_model=PredictResponse)
def predict(employee: EmployeeInput):
    _ensure_model()
    frame = pd.DataFrame([employee.model_dump()])
    result = _predict_frame(frame)[0]
    return PredictResponse(**result)


@app.post('/predict/batch', response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest):
    _ensure_model()
    rows = [i.model_dump() for i in body.employees]
    frame = pd.DataFrame(rows)
    results = _predict_frame(frame)
    items = [BatchPredictItem(index=i, **row) for i, row in enumerate(results)]
    return BatchPredictResponse(predictions=items, count=len(items))
