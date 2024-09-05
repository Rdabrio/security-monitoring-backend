import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.optimizers import Adam
import json

# Cargar el archivo JSON completo
json_file_path = '../../../app/data/catalonia_data.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

def preparar_datos_completo(data):
    comunidades = [item["ABP"] for item in data["datos_ABP"]]
    años = [item["año"] for item in data["datos_ABP"]]
    numeros = [item["numero"] for item in data["datos_ABP"]]
    
    # One-Hot Encoding para las comunidades autónomas
    encoder = OneHotEncoder(sparse_output=False)
    comunidades_encoded = encoder.fit_transform(np.array(comunidades).reshape(-1, 1))
    
    # Escalado del número de crímenes usando MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))
    numeros_scaled = scaler.fit_transform(np.array(numeros).reshape(-1, 1))
    
    # Guardar los valores de normalización
    min_value = scaler.data_min_
    max_value = scaler.data_max_
    np.save('../../../app/data/abp/min_value_NC.npy', min_value)
    np.save('../../../app/data/abp/max_value_NC.npy', max_value)
    np.save('../../../app/data/abp/encoder_NC.npy', encoder.categories_)
    
    # Ajustar las dimensiones para la concatenación
    X = np.concatenate([comunidades_encoded[:-1], numeros_scaled[:-1]], axis=1)
    y = numeros_scaled[1:]
    
    # Redimensionar para LSTM/GRU
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # 1 timestep por entrada
    
    return X, y, encoder

def entrenar_modelo_lstm_gru(X, y):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=128, return_sequences=True))  # Añadir más capas
    model.add(Dropout(0.3))
    model.add(LSTM(units=64))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X, y, epochs=300, batch_size=32, verbose=2)
    
    # Guardar el modelo entrenado
    model.save('../../../app/saved_models/abp/numbercrimes.h5')

X, y, encoder = preparar_datos_completo(data)
entrenar_modelo_lstm_gru(X, y)
