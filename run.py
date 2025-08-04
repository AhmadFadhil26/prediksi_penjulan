import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from datetime import datetime
import math

DATA_FILE = 'dataset.csv'
MODEL_FILE = 'model.pkl'

st.set_page_config(page_title="Prediksi Penjualan", layout="wide")

# === Navigasi ===
menu = st.sidebar.selectbox("📂 Menu", [
    "🏠 Resume", "📊 Lihat Dataset", "📈 Training Model",
    "🔮 Prediksi Penjualan", "📉 Evaluasi Model",
    "📝 Input Data Baru", "⚖️ Bandingkan Model"
])

# === Helper ===
def load_data():
    df = pd.read_csv(DATA_FILE)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df['Tahun'] = df['Tanggal'].dt.year
    df['Bulan_Angka'] = df['Tanggal'].dt.month
    df['Hari'] = df['Tanggal'].dt.day
    df['Bulan'] = df['Tanggal'].dt.strftime('%B')
    df.drop(columns=['Tanggal'], inplace=True)
    return df

def preprocess(df):
    numeric_cols = ['Harga Satuan', 'Jumlah Unit', 'Total Harga', 'Diskon', 'Stok Awal', 'Tahun', 'Bulan_Angka', 'Hari']
    categorical_cols = ['Kategori', 'Jenis', 'Musim/Event', 'Jenis Transaksi', 'Bulan']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df, numeric_cols, categorical_cols

# === Page: Resume ===
if menu == "🏠 Resume":
    st.title("📊 Ringkasan Data")
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        st.write(f"Jumlah data: {len(df)}")
        st.write("Data terakhir:")
        st.write(df.tail(1))
    else:
        st.warning("Dataset belum tersedia.")

# === Page: Lihat Dataset ===
elif menu == "📊 Lihat Dataset":
    st.title("📑 Dataset Penjualan")
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        st.dataframe(df)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "dataset.csv")
    else:
        st.warning("File dataset.csv belum tersedia.")

# === Page: Training Model ===
elif menu == "📈 Training Model":
    st.title("🚀 Training Model MLP")
    df = load_data()
    df, num_cols, cat_cols = preprocess(df)

    X = df.drop(columns=['Target Penjualan'])
    y = df['Target Penjualan']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_FILE)
    score = pipeline.score(X_test, y_test)
    st.success(f"Model berhasil ditraining. R² score: {score:.2f}")

# === Page: Prediksi ===
elif menu == "🔮 Prediksi Penjualan":
    st.title("🔮 Prediksi Penjualan")
    if not os.path.exists(MODEL_FILE):
        st.warning("Model belum ditraining.")
    else:
        with st.form("form_prediksi"):
            tanggal = st.date_input("Tanggal")
            kategori = st.selectbox("Kategori", ["Boneka", "Bantal"])
            jenis = st.text_input("Jenis")
            harga = st.number_input("Harga Satuan", 0)
            jumlah = st.number_input("Jumlah Unit", 0)
            total = st.number_input("Total Harga", 0)
            diskon = st.number_input("Diskon", 0)
            musim = st.selectbox("Musim/Event", ["Hari Anak", "Libur Sekolah", "Tahun Baru", "Promo"])
            stok_awal = st.number_input("Stok Awal", 0)
            transaksi = st.selectbox("Jenis Transaksi", ["Debit", "Kredit", "Tunai"])
            submit = st.form_submit_button("Prediksi")

            if submit:
                df = pd.DataFrame([{
                    'Tanggal': tanggal,
                    'Kategori': kategori,
                    'Jenis': jenis,
                    'Harga Satuan': harga,
                    'Jumlah Unit': jumlah,
                    'Total Harga': total,
                    'Diskon': diskon,
                    'Musim/Event': musim,
                    'Stok Awal': stok_awal,
                    'Jenis Transaksi': transaksi
                }])
                df['Tanggal'] = pd.to_datetime(df['Tanggal'])
                df['Hari'] = df['Tanggal'].dt.day
                df['Bulan'] = df['Tanggal'].dt.strftime('%B')
                df['Bulan_Angka'] = df['Tanggal'].dt.month
                df['Tahun'] = df['Tanggal'].dt.year
                df.drop(columns=['Tanggal'], inplace=True)

                model = joblib.load(MODEL_FILE)
                pred = model.predict(df)[0]
                pred = math.ceil(pred)

                st.success(f"📈 Prediksi penjualan: {pred} unit")

                if stok_awal < pred:
                    st.warning(f"Stok kurang! Tambahkan minimal {pred - stok_awal} unit.")
                elif stok_awal > pred * 1.5:
                    st.info(f"Stok terlalu banyak! Ideal: maks {math.ceil(pred * 1.5)} unit.")
                else:
                    st.success("Stok sudah mencukupi.")

# === Page: Evaluasi ===
elif menu == "📉 Evaluasi Model":
    st.title("📉 Evaluasi Model MLP")
    if not os.path.exists(MODEL_FILE):
        st.warning("Model belum ditraining.")
    else:
        df = load_data()
        df, num_cols, cat_cols = preprocess(df)

        X = df.drop(columns=['Target Penjualan'])
        y = df['Target Penjualan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = joblib.load(MODEL_FILE)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.metric("MSE", f"{mse:.2f}")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("R² Score", f"{r2:.2f}")

        st.subheader("📊 Grafik: Aktual vs Prediksi")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.values[:50], label="Aktual", marker='o')
        ax.plot(y_pred[:50], label="Prediksi", marker='x')
        ax.legend()
        st.pyplot(fig)

# === Page: Input Data Baru ===
elif menu == "📝 Input Data Baru":
    st.title("📝 Input Data Manual")
    with st.form("form_input"):
        tanggal = st.date_input("Tanggal")
        kategori = st.selectbox("Kategori", ["Boneka", "Bantal"])
        jenis = st.text_input("Jenis")
        harga = st.number_input("Harga Satuan", 0)
        jumlah = st.number_input("Jumlah Unit", 0)
        total = st.number_input("Total Harga", 0)
        diskon = st.number_input("Diskon", 0)
        musim = st.selectbox("Musim/Event", ["Hari Anak", "Libur Sekolah", "Tahun Baru", "Promo"])
        stok_awal = st.number_input("Stok Awal", 0)
        transaksi = st.selectbox("Jenis Transaksi", ["Debit", "Kredit", "Tunai"])
        target = st.number_input("Target Penjualan", 0)
        submit = st.form_submit_button("Simpan")

        if submit:
            row = {
                'Tanggal': tanggal,
                'Kategori': kategori,
                'Jenis': jenis,
                'Harga Satuan': harga,
                'Jumlah Unit': jumlah,
                'Total Harga': total,
                'Diskon': diskon,
                'Musim/Event': musim,
                'Stok Awal': stok_awal,
                'Jenis Transaksi': transaksi,
                'Bulan': tanggal.month,
                'Target Penjualan': target
            }

            df = pd.DataFrame([row])
            if os.path.exists(DATA_FILE):
                df_old = pd.read_csv(DATA_FILE)
                df = pd.concat([df_old, df], ignore_index=True)

            df.to_csv(DATA_FILE, index=False)
            st.success("✅ Data berhasil ditambahkan.")

# === Page: Bandingkan Model ===
elif menu == "⚖️ Bandingkan Model":
    st.title("⚖️ Perbandingan Model MLP")
    if st.button("🔁 Jalankan Ulang Perbandingan"):
        subprocess.run(["python", "compare_mlp_configs.py"])
        st.success("Perbandingan model diperbarui.")

    if os.path.exists("hasil_perbandingan_mlp.csv"):
        df = pd.read_csv("hasil_perbandingan_mlp.csv")
        st.dataframe(df)
    else:
        st.warning("File hasil_perbandingan_mlp.csv belum tersedia.")
