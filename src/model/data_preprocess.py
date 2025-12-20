from datetime import datetime
from meteostat import Point, Hourly
import pandas as pd

#The location of Charlottesville
location = Point(38.03, -78.48)


data = Hourly(location, start=datetime(2024, 1, 1), end=datetime(2024, 12, 1))
df = data.fetch()

#Clean the data, only save temperature, humidity and air pressure
df = df[['temp', 'rhum', 'pres']]
df.dropna(inplace=True)

print(df.head())
df.dropna(inplace=True)

import numpy as np

WINDOW = 24   #Past 24 hours
HORIZON = 1   #Predict the temperature for next one hour

def make_supervised(series, window=24, horizon=1):
    temps = series.values.astype('float32')
    X, y = [], []
    for i in range(len(temps) - window - horizon + 1):
        X.append(temps[i:i+window])
        y.append(temps[i+window+horizon-1])
    return np.array(X), np.array(y)

X, y = make_supervised(df['temp'], window=WINDOW, horizon=HORIZON)
print(X.shape, y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(WINDOW,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)                  #Output the predicted T
])

model.compile(
    optimizer='adam',
    loss='mse',      
    metrics=['mae']   
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1
)

test_loss, test_mae = model.evaluate(X_test, y_test)
print("Test MAE (°C):", test_mae)

