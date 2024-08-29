import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import json

# Cargar el modelo entrenado
model = load_model('c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/saved_models/ccaa/numbercrimes.h5')

# Cargar los valores de normalización y el encoder
min_value = np.load('c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/data/ccaa/min_value_NC.npy')
max_value = np.load('c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/data/ccaa/max_value_NC.npy')

# Cargar el archivo JSON completo y el OneHotEncoder
json_file_path = 'c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/data/ccaa_data.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

comunidades = [item["comunidad"]["nombre"] for item in data["datos_comunidades"]]
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.array(comunidades).reshape(-1, 1))

def predecir_numero_crimenes_anual(comunidad, año):
    # Filtrar los datos para la comunidad y el año específicos
    datos_filtrados = [
        item for item in data["datos_comunidades"]
        if item["comunidad"]["nombre"] == comunidad and item["año"] == año
    ]
    
    if not datos_filtrados:
        raise ValueError("Datos no encontrados para la comunidad y año especificados.")
    
    # One-Hot Encoding para la comunidad autónoma
    comunidad_encoded = encoder.transform([[comunidad]])
    
    df = pd.DataFrame(datos_filtrados)
    
    # Escalar el número de crímenes para la predicción usando MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(np.array([[min_value], [max_value]]).reshape(-1, 1))
    scaled_data = scaler.transform(df['numero'].values.reshape(-1, 1))

    # Preparar la entrada para el modelo
    X = np.concatenate([comunidad_encoded, scaled_data[-1].reshape(1, -1)], axis=1)
    X = X.reshape(1, 1, X.shape[1])  # 1 timestep y n características
    
    # Hacer la predicción
    predicted_value = model.predict(X)
    
    # Desescalar los valores predichos
    predicted_value_descaled = predicted_value * (max_value - min_value) + min_value
    
    return predicted_value_descaled.flatten().tolist()

# Ejemplo de uso para predicción de un año específico
comunidad = "Cataluña"
año = 2022
prediccion = predecir_numero_crimenes_anual(comunidad, año)
print(f"Predicción de crímenes en {año} para {comunidad}: {prediccion}")
