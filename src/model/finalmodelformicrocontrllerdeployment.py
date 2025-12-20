import numpy as np
import pandas as pd
import tensorflow as tf

# ---------- CONFIG ----------
CSV_FILES = [
    "day1.csv",

]

SAMPLE_SECONDS = 10
WINDOW_MINUTES = 30
HORIZON_MINUTES = 30

WINDOW = int(WINDOW_MINUTES * 60 / SAMPLE_SECONDS)     # 180
HORIZON = int(HORIZON_MINUTES * 60 / SAMPLE_SECONDS)   # 180

FEATURES = ["temp", "rhum", "pres"]

# ---------- LOAD ----------
dfs = []
for f in CSV_FILES:
    df = pd.read_csv(f)
    df = df.rename(columns={
    "Temperature (C)": "temp",
    "Humidity (%)": "rhum",
    "Pressure (hPa)": "pres"
})
    df = df[FEATURES].dropna()
    dfs.append(df)

df = pd.concat(dfs, axis=0).reset_index(drop=True)

# ---------- SUPERVISED ----------
vals = df.values.astype("float32")
X, y = [], []

for i in range(len(vals) - WINDOW - HORIZON + 1):
    X.append(vals[i:i+WINDOW])
    y.append(vals[i+WINDOW+HORIZON-1, 0])  # future temp

X = np.array(X)
y = np.array(y)

# ---------- NORMALIZE ----------
x_mean = X.mean(axis=(0,1), keepdims=True)
x_std  = X.std(axis=(0,1), keepdims=True) + 1e-6
y_mean = y.mean()
y_std  = y.std() + 1e-6

Xn = (X - x_mean) / x_std
yn = (y - y_mean) / y_std

# ---------- MODEL ----------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(Xn, yn, epochs=100, batch_size=16, shuffle=False)

# ---------- INT8 EXPORT ----------
def rep_data():
    for i in range(len(Xn)):
        yield [Xn[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite = converter.convert()
open("model_int8.tflite", "wb").write(tflite)

# ---------- SAVE NORMALIZATION ----------
np.savez(
    "norm_params.npz",
    x_mean=x_mean.flatten(),
    x_std=x_std.flatten(),
    y_mean=y_mean,
    y_std=y_std,
    window=WINDOW,
    horizon=HORIZON
)
