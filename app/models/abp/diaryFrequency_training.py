import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import json

# Paso 1: Cargar y Preparar los Datos desde un JSON Local

# Ruta del archivo JSON local
json_file_path = '../../../app/data/catalonia_data.json'

# Cargar los datos desde el archivo JSON
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convertir los datos a un DataFrame
df = pd.DataFrame(data["datos_ABP"])

# Paso 2: Filtrar y Procesar los Datos

# Convertir el número anual en frecuencias diarias
df["numero_diario"] = df["numero"] / 365

# Crear un DataFrame para almacenar los datos interpolados
df_diario = pd.DataFrame()

for index, row in df.iterrows():
    # Generar un rango de fechas para cada año
    fechas = pd.date_range(start=f'{row["año"]}-01-01', end=f'{row["año"]}-12-31', freq='D')
    
    # Crear un DataFrame temporal con las fechas y el número diario
    temp_df = pd.DataFrame({
        "fecha": fechas,
        "numero": row["numero_diario"]
    })
    
    # Concatenar los datos diarios
    df_diario = pd.concat([df_diario, temp_df], ignore_index=True)

# Paso 3: Preprocesamiento de los Datos

# Normalizar los datos (importante para la red neuronal)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_diario['numero'].values.reshape(-1, 1))

# Preparar los datos para la red neuronal
window_size = 7  # Usamos una ventana de 7 días para predecir el siguiente día
X, y = [], []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Paso 4: Crear y Entrenar el Modelo

# Crear el modelo CNN 1D + LSTM
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X, y, epochs=20, batch_size=32, verbose=2)

# Paso 5: Guardar el Modelo y los Valores de Normalización

# Guardar el modelo entrenado en un archivo .h5
model.save('../../../app/saved_models/abp/diaryfrequancy.h5')

# Guardar los valores de normalización (min y max) para usarlos en las predicciones
min_value = df_diario['numero'].min()
max_value = df_diario['numero'].max()
np.save('../../../app/data/abp/min_value_DF.npy', min_value)
np.save('../../../app/data/abp/max_value_DF.npy', max_value)

