import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
import json

# Cargar el archivo JSON completo
json_file_path = 'c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/data/ccaa_data.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Preparar los datos para el entrenamiento
def preparar_datos_completo(data):
    datos = []
    for item in data["datos_comunidades"]:
        datos.append({"año": item["año"], "numero": item["numero"] / 365})
    
    df = pd.DataFrame(datos)
    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_escalados = scaler.fit_transform(df['numero'].values.reshape(-1, 1))
    
    # Guardar los valores de normalización
    np.save('c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/data/ccaa/min_value_LT.npy', scaler.data_min_)
    np.save('c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/data/ccaa/max_value_LT.npy', scaler.data_max_)
    
    # Preparar los datos en secuencias para el modelo GRU
    X, y = [], []
    window_size = 7  # Usamos 7 días para predecir el siguiente valor
    for i in range(window_size, len(datos_escalados)):
        X.append(datos_escalados[i-window_size:i, 0])
        y.append(datos_escalados[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y

# Crear y entrenar el modelo GRU
def entrenar_modelo_gru(X, y):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(GRU(units=50))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=100, batch_size=32, verbose=2)
    
    # Guardar el modelo entrenado
    model.save('c:/Users/Dabrio/Desktop/Proyectos/security-monitoring-backend-def/security-monitoring-backend/app/saved_models/ccaa/longterm.h5')

# Ejemplo de entrenamiento con todos los datos
X, y = preparar_datos_completo(data)
entrenar_modelo_gru(X, y)
