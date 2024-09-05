import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import json

# Cargar el archivo JSON completo
json_file_path = '../../../app/data/catalonia_data.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Preparar los datos para el entrenamiento
def preparar_datos_completo(data):
    comunidades = []
    numeros = []
    for item in data["datos_ABP"]:
        comunidades.append(item["ABP"])
        numeros.append(item["numero"] / 365)
    
    df = pd.DataFrame({
        'comunidad': comunidades,
        'numero_diario': numeros
    })

    # One-Hot Encoding para las comunidades autónomas
    encoder = OneHotEncoder(sparse_output=False)
    comunidades_encoded = encoder.fit_transform(df['comunidad'].values.reshape(-1, 1))

    scaler = MinMaxScaler(feature_range=(0, 1))
    numeros_scaled = scaler.fit_transform(df['numero_diario'].values.reshape(-1, 1))
    
    # Guardar los valores de normalización
    np.save('../../../app/data/abp/min_value_LT.npy', scaler.data_min_)
    np.save('../../../app/data/abp/max_value_LT.npy', scaler.data_max_)
    np.save('../../../app/data/abp/encoder_LT.npy', encoder.categories_)
    
    # Preparar los datos en secuencias para el modelo GRU
    X_combined = np.concatenate([comunidades_encoded, numeros_scaled], axis=1)

    X, y = [], []
    window_size = 7  # Usamos 7 días para predecir el siguiente valor
    for i in range(window_size, len(X_combined)):
        X.append(X_combined[i-window_size:i])
        y.append(numeros_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    
    return X, y, encoder

# Crear y entrenar el modelo GRU
def entrenar_modelo_gru(X, y):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(GRU(units=50))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=100, batch_size=32, verbose=2)
    
    # Guardar el modelo entrenado
    model.save('../../../app/saved_models/abp/longterm.h5')

# Ejemplo de entrenamiento con todos los datos
X, y, encoder = preparar_datos_completo(data)
entrenar_modelo_gru(X, y)
