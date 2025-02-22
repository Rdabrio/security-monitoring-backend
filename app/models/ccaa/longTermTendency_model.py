import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo entrenado
model = load_model(os.path.join(base_dir,'../../../app/saved_models/ccaa/longterm.h5'))

# Cargar los valores de normalización
min_value = np.load(os.path.join(base_dir,'../../../app/data/ccaa/min_value_LT.npy'))
max_value = np.load(os.path.join(base_dir,'../../../app/data/ccaa/max_value_LT.npy'))
encoder_categories = np.load(os.path.join(base_dir,'../../../app/data/ccaa/encoder_LT.npy'), allow_pickle=True)
encoder_categories = encoder_categories.tolist()

# Cargar el archivo JSON completo
json_file_path = os.path.join(base_dir,'../../../app/data/ccaa_data.json')
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

encoder = OneHotEncoder(sparse_output=False, categories=encoder_categories)
encoder.fit(np.array(encoder_categories).reshape(-1, 1))

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

    comunidad_encoded = encoder.transform([[comunidad]])

    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_escalados = scaler.fit_transform(df['numero_diario'].values.reshape(-1, 1))

    # Preparar los datos para la predicción (última ventana)
    window_size = 7
    X_combined = np.concatenate([comunidad_encoded.repeat(window_size, axis=0), datos_escalados[-window_size:]], axis=1)
    X = X_combined.reshape(1, window_size, X_combined.shape[1])
    
    # Hacer predicciones
    predicted_values = model.predict(X)
    
    # Desnormalizar los valores predichos
    predicted_values_descaled = predicted_values * (max_value - min_value) + min_value
    
    return predicted_values_descaled.flatten().tolist()
