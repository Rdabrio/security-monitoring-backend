import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo entrenado
model = load_model(os.path.join(base_dir,'../../../app/saved_models/abp/diaryfrequency.h5'))

# Cargar los valores de normalización
min_value = np.load(os.path.join(base_dir,'../../../app/data/abp/min_value_DF.npy'))
max_value = np.load(os.path.join(base_dir,'../../../app/data/abp/max_value_DF.npy'))

# Cargar el archivo JSON completo
json_file_path = os.path.join(base_dir,'../../../app/data/catalonia_data.json')
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

def predecir(comunidad, año):
    # Filtrar los datos
    datos_filtrados = [
        item for item in data["datos_ABP"]
        if item["ABP"] == comunidad and item["año"] == año
    ]
    
    if not datos_filtrados:
        raise ValueError("Datos no encontrados para la comunidad y año especificados.")
    
    # Preprocesar los datos
    df = pd.DataFrame(datos_filtrados)
    df['numero_diario'] = df['numero'] / 365
    datos_escalados = MinMaxScaler(feature_range=(0, 1)).fit_transform(df['numero_diario'].values.reshape(-1, 1))

    # Hacer predicciones
    window_size = 7
    entrada_actual = datos_escalados[-window_size:].reshape(1, window_size, 1)
    predicciones_futuras = []
    for _ in range(10):
        prediccion = model.predict(entrada_actual)
        predicciones_futuras.append(prediccion[0, 0])
        entrada_actual = np.append(entrada_actual[:, 1:, :], prediccion.reshape(1, 1, 1), axis=1)

    # Desnormalizar
    predicciones_futuras = np.array(predicciones_futuras).reshape(-1, 1)
    predicciones_desnormalizadas = predicciones_futuras * (max_value - min_value) + min_value
    return predicciones_desnormalizadas.flatten().tolist()


