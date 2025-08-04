import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# === 1. Load dataset ===
data_path = 'data/dataset.csv'
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

# === 2. Preprocessing kolom waktu seperti di run.py ===
df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
df['Tahun'] = df['Tanggal'].dt.year
df['Bulan_Angka'] = df['Tanggal'].dt.month
df['Hari'] = df['Tanggal'].dt.day
df['Bulan'] = df['Tanggal'].dt.strftime('%B')
df.drop(columns=['Tanggal'], inplace=True)

# === 3. Tentukan kolom target dan fitur ===
TARGET_KOLOM = 'Target Penjualan'

if TARGET_KOLOM not in df.columns:
    raise ValueError(f"Kolom target '{TARGET_KOLOM}' tidak ditemukan.")

X = df.drop(columns=[TARGET_KOLOM])
y = df[TARGET_KOLOM]

# === 4. Tentukan kolom numerik dan kategorik secara eksplisit (seperti run.py) ===
numeric_cols = ['Harga Satuan', 'Jumlah Unit', 'Total Harga', 'Diskon', 'Stok Awal', 'Tahun', 'Bulan_Angka', 'Hari']
categorical_cols = ['Kategori', 'Jenis', 'Musim/Event', 'Jenis Transaksi', 'Bulan']

# Konversi numerik dan hilangkan NaN
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.dropna()
y = y[X.index]

# === 5. Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 6. Preprocessor identik ===
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# === 7. Konfigurasi model ===
model_configs = [
    {
        'name': 'Default',
        'params': {}
    },
    {
        'name': 'Model Config dari run.py',
        'params': {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 1000
        }
    },
    {
        'name': 'Model Config Contoh Lain 1',
        'params': {
            'hidden_layer_sizes': (128, 64, 32),
            'activation': 'tanh',
            'solver': 'lbfgs',
            'alpha': 0.001,
            'max_iter': 500
        }
    },
    {
        'name': 'Model Config Contoh Lain 2',
        'params': {
            'hidden_layer_sizes': (64,),
            'activation': 'relu',
            'solver': 'sgd',
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.005,
            'momentum': 0.9,
            'max_iter': 500
        }
    }
]

# === 8. Training dan evaluasi ===
results = []

for config in model_configs:
    print(f"Training model: {config['name']}")
    model = MLPRegressor(random_state=42, **config['params'])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    results.append({
        'Model': config['name'],
        'R2 Score': round(r2, 4),
        'MSE': round(mse, 4)
    })

# === 9. Simpan hasil ===
results_df = pd.DataFrame(results)
print("\nHasil Perbandingan Model:")
print(results_df)
results_df.to_csv('hasil_perbandingan_mlp.csv', index=False)
print("\nHasil disimpan ke 'hasil_perbandingan_mlp.csv'")
