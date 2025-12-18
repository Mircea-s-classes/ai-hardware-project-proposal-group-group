from datetime import datetime
import numpy as np
import pandas as pd

# ---------- 1) Config ----------
RESAMPLE_MIN = 5            # unify both datasets to 5-minute bins
WINDOW_STEPS = 12           # 12 * 5min = 60 minutes history
HORIZON_STEPS = 12          # predict 12 * 5min = 60 minutes ahead (next hour)
FEATURE_COLS = ["temp", "rhum", "pres"]
TARGET_COL = "temp"

# Your uploaded file (change path if needed)
ARDUINO_CSV = "Next Hour Temperature Live Data - 12_17 1pm-6_15pm.csv"

# Meteostat location: Charlottesville
LAT, LON = 38.03, -78.48
WEB_START = datetime(2024, 1, 1)
WEB_END   = datetime(2024, 12, 1)

# ---------- 2) Helpers ----------
def make_supervised_multivariate(df, feature_cols, target_col, window_steps, horizon_steps):
    """
    df: time-indexed dataframe, regularly sampled
    X shape: (N, window_steps, F)
    y shape: (N,)
    """
    feat = df[feature_cols].values.astype("float32")
    tgt  = df[target_col].values.astype("float32")

    X, y = [], []
    T = len(df)
    for i in range(T - window_steps - horizon_steps + 1):
        X.append(feat[i:i+window_steps, :])
        y.append(tgt[i+window_steps+horizon_steps-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def zscore_fit(X, y, eps=1e-6):
    x_mean = X.mean(axis=(0, 1), keepdims=True)
    x_std  = X.std(axis=(0, 1), keepdims=True) + eps
    y_mean = float(y.mean())
    y_std  = float(y.std() + eps)
    return x_mean, x_std, y_mean, y_std

def zscore_apply_X(X, x_mean, x_std):
    return (X - x_mean) / x_std

def zscore_apply_y(y, y_mean, y_std):
    return (y - y_mean) / y_std

def inv_zscore_y(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean

# ---------- 3) Load Meteostat (historical pretrain) ----------
def load_web_data():
    # import here so your script still runs even if meteostat isn't installed
    from meteostat import Point, Hourly

    location = Point(LAT, LON)
    data = Hourly(location, start=WEB_START, end=WEB_END)
    df = data.fetch()

    # Keep only needed columns
    df = df[["temp", "rhum", "pres"]].dropna()

    # Make sure time index exists and is datetime
    df.index = pd.to_datetime(df.index)

    # Resample hourly -> every RESAMPLE_MIN minutes with interpolation
    rule = f"{RESAMPLE_MIN}min"
    df = df.resample(rule).interpolate(method="time").dropna()

    return df

# ---------- 4) Load Arduino (fine-tune) ----------
def load_arduino_data(csv_path):
    df = pd.read_csv(csv_path)

    # Rename columns to match web naming
    df = df.rename(columns={
        "Temperature (C)": "temp",
        "Humdity (%)": "rhum",
        "Pressure (hPa)": "pres",
    })

    # Build a time index from relative ms (start at 0)
    # This gives us a proper time axis for resampling.
    if "Relative Time (ms)" not in df.columns:
        raise ValueError("Arduino CSV must contain 'Relative Time (ms)' column.")

    df["t"] = pd.to_timedelta(df["Relative Time (ms)"], unit="ms")
    df = df.set_index("t")

    # Keep only features, drop missing
    df = df[FEATURE_COLS].dropna()

    # Resample to same granularity as web (5-min mean)
    rule = f"{RESAMPLE_MIN}min"
    df = df.resample(rule).mean().dropna()

    return df

# ---------- 5) Build model ----------
def build_small_mlp(window_steps, num_features):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_steps, num_features)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    return model

# ---------- 6) Main: Pretrain -> Fine-tune -> Export int8 ----------
def main():
    import tensorflow as tf

    # Load datasets
    df_web = load_web_data()
    df_ard = load_arduino_data(ARDUINO_CSV)

    # Create supervised sets
    X_web, y_web = make_supervised_multivariate(df_web, FEATURE_COLS, TARGET_COL, WINDOW_STEPS, HORIZON_STEPS)
    X_ard, y_ard = make_supervised_multivariate(df_ard, FEATURE_COLS, TARGET_COL, WINDOW_STEPS, HORIZON_STEPS)

    print("Web samples:", X_web.shape, y_web.shape)
    print("Arduino samples:", X_ard.shape, y_ard.shape)

    if len(X_ard) < 10:
        print("Arduino fine-tune samples are very few after resampling/windowing.")
        print("Try reducing WINDOW_STEPS/HORIZON_STEPS or collecting longer device data.")
        return

    # Split web: train/test (time order)
    split_web = int(len(X_web) * 0.8)
    X_web_train, X_web_test = X_web[:split_web], X_web[split_web:]
    y_web_train, y_web_test = y_web[:split_web], y_web[split_web:]

    # Split arduino: fine-tune train / arduino test (time order)
    split_ard = int(len(X_ard) * 0.8)
    X_ard_train, X_ard_test = X_ard[:split_ard], X_ard[split_ard:]
    y_ard_train, y_ard_test = y_ard[:split_ard], y_ard[split_ard:]

    # Normalize using WEB TRAIN stats (important!)
    x_mean, x_std, y_mean, y_std = zscore_fit(X_web_train, y_web_train)

    X_web_train_n = zscore_apply_X(X_web_train, x_mean, x_std)
    y_web_train_n = zscore_apply_y(y_web_train, y_mean, y_std)
    X_web_test_n  = zscore_apply_X(X_web_test,  x_mean, x_std)
    y_web_test_n  = zscore_apply_y(y_web_test,  y_mean, y_std)

    X_ard_train_n = zscore_apply_X(X_ard_train, x_mean, x_std)
    y_ard_train_n = zscore_apply_y(y_ard_train, y_mean, y_std)
    X_ard_test_n  = zscore_apply_X(X_ard_test,  x_mean, x_std)
    y_ard_test_n  = zscore_apply_y(y_ard_test,  y_mean, y_std)

    # Build + pretrain
    model = build_small_mlp(WINDOW_STEPS, len(FEATURE_COLS))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

    print("\n--- Pretraining on web data ---")
    model.fit(X_web_train_n, y_web_train_n, epochs=20, batch_size=64, validation_split=0.1, shuffle=False, verbose=1)

    web_loss, web_mae = model.evaluate(X_web_test_n, y_web_test_n, verbose=0)
    print("Web test MAE (normalized):", float(web_mae))
    print("Web test MAE (°C approx):", float(web_mae) * y_std)

    # Fine-tune on Arduino (small LR, few epochs)
    print("\n--- Fine-tuning on Arduino data (5 hours) ---")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
    model.fit(X_ard_train_n, y_ard_train_n, epochs=8, batch_size=16, validation_split=0.1, shuffle=False, verbose=1)

    ard_loss, ard_mae = model.evaluate(X_ard_test_n, y_ard_test_n, verbose=0)
    print("Arduino test MAE (normalized):", float(ard_mae))
    print("Arduino test MAE (°C approx):", float(ard_mae) * y_std)

    # Save keras model
    model.save("temp_model_finetuned.keras")
    print("Saved keras model: temp_model_finetuned.keras")

    # Export int8 TFLite (for Arduino)
    def representative_data_gen():
        # Use a mix of web + arduino samples for calibration
        reps = np.concatenate([X_web_train_n[:300], X_ard_train_n[:300]], axis=0)
        for i in range(len(reps)):
            yield [reps[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8 = converter.convert()
    with open("model_int8.tflite", "wb") as f:
        f.write(tflite_int8)
    print("Saved TFLite int8 model: model_int8.tflite")

    # Save normalization params (needed on device!)
    np.savez("norm_params.npz",
             x_mean=x_mean, x_std=x_std,
             y_mean=np.array([y_mean], dtype=np.float32),
             y_std=np.array([y_std], dtype=np.float32),
             resample_min=np.array([RESAMPLE_MIN], dtype=np.int32),
             window_steps=np.array([WINDOW_STEPS], dtype=np.int32),
             horizon_steps=np.array([HORIZON_STEPS], dtype=np.int32))

    print("Saved normalization params: norm_params.npz")
    print("\nDone.")

if __name__ == "__main__":
    main()
