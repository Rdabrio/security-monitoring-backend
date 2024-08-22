import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Supongamos que los datos están en un DataFrame de Pandas
data = pd.DataFrame({
    'año': [2013, 2014, 2015, 2016, 2017],
    'numero': [13305, 13100, 12027, 12021, 12244]
})

# Preprocesamiento de los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['numero'].values.reshape(-1, 1))

# Preparar los datos para LSTM
X, y = [], []
for i in range(1, len(scaled_data)):
    X.append(scaled_data[i-1:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=1, verbose=2)

# Hacer predicciones
predicted_value = model.predict(X[-1].reshape(1, 1, 1))
predicted_value = scaler.inverse_transform(predicted_value)
print(predicted_value)
