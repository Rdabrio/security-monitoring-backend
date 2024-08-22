from fastapi import FastAPI
from app.api.ccaa import Predict_DF

app = FastAPI()

# Incluir las rutas de los diferentes modelos
app.include_router(Predict_DF.router, prefix="/api/ccaa/DF", tags=["Model 1"])
#app.include_router(predict_model2.router, prefix="/api/v1/model2", tags=["Model 2"])
#app.include_router(predict_model3.router, prefix="/api/v1/model3", tags=["Model 3"])

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de predicciones de delitos"}
