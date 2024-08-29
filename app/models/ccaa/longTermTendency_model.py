import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json

# Cargar el modelo entrenado
model = load_model('c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/saved_models/ccaa/longterm.h5')

# Cargar los valores de normalización
min_value = np.load('c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/data/ccaa/min_value_LT.npy')
max_value = np.load('c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/data/ccaa/max_value_LT.npy')

# Cargar el archivo JSON completo
json_file_path = 'c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/data/ccaa_data.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

def predecir_tendencia(comunidad, año):
    # Filtrar los datos para la comunidad y el año específicos
    datos_filtrados = [
        item for item in data["datos_comunidades"]
        if item["comunidad"]["nombre"] == comunidad and item["año"] == año
    ]
    
    if not datos_filtrados:
        raise ValueError("Datos no encontrados para la comunidad y año especificados.")
    
    # Preprocesar los datos para predicción
    df = pd.DataFrame(datos_filtrados)
    df['numero_diario'] = df['numero'] / 365
    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_escalados = scaler.fit_transform(df['numero_diario'].values.reshape(-1, 1))

    # Preparar los datos para la predicción (última ventana)
    window_size = 7
    X = datos_escalados[-window_size:].reshape(1, window_size, 1)
    
    # Hacer predicciones
    predicted_values = model.predict(X)
    
    # Desnormalizar los valores predichos
    predicted_values_descaled = predicted_values * (max_value - min_value) + min_value
    
    return predicted_values_descaled.flatten().tolist()

# Ejemplo de uso para predicción
comunidad = "Cataluña"
año = 2020
tendencia_futura = predecir_tendencia(comunidad, año)
print(tendencia_futura)
