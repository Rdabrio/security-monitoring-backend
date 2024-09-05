from fastapi import FastAPI
from app.api.ccaa import Predict_DF, Predict_LT, Predict_NC
from app.api.abp import Predict_DF as Predict_DF_ABP, Predict_LT as Predict_LT_ABP, Predict_NC as Predict_NC_ABP

app = FastAPI()

app.include_router(Predict_DF.router, prefix="/api/ccaa/DF", tags=["Model 1"])
app.include_router(Predict_LT.router, prefix="/api/ccaa/LT", tags=["Model 2"])
app.include_router(Predict_NC.router, prefix="/api/ccaa/NC", tags=["Model 3"])
app.include_router(Predict_DF_ABP.router, prefix="/api/abp/DF", tags=["Model 4"])
app.include_router(Predict_LT_ABP.router, prefix="/api/abp/LT", tags=["Model 5"])
app.include_router(Predict_NC_ABP.router, prefix="/api/abp/NC", tags=["Model 6"])

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de predicciones de delitos"}