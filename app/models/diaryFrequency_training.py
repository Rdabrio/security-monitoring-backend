import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Datos de ejemplo
data = [
    {"comunidad": {"nombre": "Andalucía"}, "delito": "Estafas", "año": 2015, "numero": 1264},
    {"comunidad": {"nombre": "Andalucía"}, "delito": "Estafas", "año": 2014, "numero": 1087},
    {"comunidad": {"nombre": "Andalucía"}, "delito": "Estafas", "año": 2013, "numero": 1038},
]

# Convertir los datos a un DataFrame
df = pd.DataFrame(data)

# Convertir el número anual en frecuencias diarias
df["numero_diario"] = df["numero"] / 365

# Crear un DataFrame vacío para almacenar los datos interpolados
df_diario = pd.DataFrame()

for index, row in df.iterrows():
    # Generar un rango de fechas para cada año
    fechas = pd.date_range(start=f'{row["año"]}-01-01', end=f'{row["año"]}-12-31', freq='D')
    
    # Crear un DataFrame temporal con las fechas y el número diario
    temp_df = pd.DataFrame({
        "fecha": fechas,
        "comunidad": row["comunidad"]["nombre"],
        "delito": row["delito"],
        "numero": row["numero_diario"]
    })
    
    # Concatenar los datos diarios
    df_diario = pd.concat([df_diario, temp_df], ignore_index=True)

# Preprocesamiento
# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_diario['numero'].values.reshape(-1, 1))

# Preparar los datos para la red neuronal
window_size = 7  # Usaremos una ventana de 7 días para predecir el siguiente día
X, y = [], []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Construir el modelo CNN 1D + LSTM
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamiento
model.fit(X, y, epochs=20, batch_size=32, verbose=2)

# Predicción de los próximos 'n' días
n_dias_futuros = 10
predicciones_futuras = []

# Usar la última ventana de datos como punto de partida
entrada_actual = scaled_data[-window_size:].reshape(1, window_size, 1)

for _ in range(n_dias_futuros):
    prediccion = model.predict(entrada_actual)
    predicciones_futuras.append(prediccion[0, 0])
    
    # Actualizar la ventana de entrada para la próxima predicción
    entrada_actual = np.append(entrada_actual[:, 1:, :], prediccion.reshape(1, 1, 1), axis=1)

# Desnormalizar las predicciones para obtener valores reales
predicciones_futuras = scaler.inverse_transform(np.array(predicciones_futuras).reshape(-1, 1))

# Mostrar predicciones
print(f"Predicciones para los próximos {n_dias_futuros} días:")
for i, prediccion in enumerate(predicciones_futuras, 1):
    print(f"Día {i}: {prediccion[0]}")
