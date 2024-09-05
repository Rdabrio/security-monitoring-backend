from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.abp import longTermTendency_model  # Importar el código relacionado con el primer modelo

router = APIRouter()

@router.get("/predict/")
def predecir(comunidad: str, año: str):
    try:
        predicciones = longTermTendency_model.predecir_tendencia(comunidad, año)
        return {"predicciones": predicciones}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
