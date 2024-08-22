from keras.layers import GRU

# Construir el modelo GRU
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(GRU(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=1, verbose=2)

# Análisis de tendencia (observación de la pendiente)
predicted_values = model.predict(X)
