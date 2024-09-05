import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo entrenado
model = load_model(os.path.join(base_dir,'../../../app/saved_models/abp/numbercrimes.h5'))

# Cargar los valores de normalización y el encoder
min_value = np.load(os.path.join(base_dir,'../../../app/data/abp/min_value_NC.npy'))
max_value = np.load(os.path.join(base_dir,'../../../app/data/abp/max_value_NC.npy'))
encoder_categories = np.load(os.path.join(base_dir,'../../../app/data/abp/encoder_NC.npy'), allow_pickle=True)
encoder_categories = encoder_categories.tolist()

# Cargar el archivo JSON completo y el OneHotEncoder
json_file_path = os.path.join(base_dir,'../../../app/data/catalonia_data.json')
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

encoder = OneHotEncoder(sparse_output=False, categories=encoder_categories)
encoder.fit(np.array(encoder_categories).reshape(-1, 1))

def predecir_numero_crimenes_anual(comunidad, año):
    print(comunidad, año)
    # Filtrar los datos para la comunidad y el año específicos
    datos_filtrados = [
        item for item in data["datos_ABP"]
        if item["ABP"] == comunidad and item["año"] == año
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