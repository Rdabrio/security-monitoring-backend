from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.ccaa import diaryFrequency_model  # Importar el código relacionado con el primer modelo

router = APIRouter()

# Definir el modelo de entrada
class PrediccionInput(BaseModel):
    comunidad: str
    año: int

@router.post("/predecir/")
def predecir(input: PrediccionInput):
    try:
        predicciones = diaryFrequency_model.predecir(input.comunidad, input.año)
        return {"predicciones": predicciones}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
