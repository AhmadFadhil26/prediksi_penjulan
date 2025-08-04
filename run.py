import streamlit as st
import pandas as pd
import joblib
import os

# Load model dan data
MODEL_PATH = "model_penjualan.pkl"
DATA_PATH = "dataset/data_penjualan.csv"

# ===== Styling Header dan Navigasi =====
def custom_style():
    st.markdown("""
        <style>
        .header {
            background-color: #0d6efd;
            padding: 20px;
            text-align: center;
            color: white;
            font-size: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .content {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }

        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }

        .logo {
            width: 40px;
            margin-right: 10px;
        }

        </style>
    """, unsafe_allow_html=True)

# ===== Komponen Header dengan Logo dan Nama Aplikasi =====
def app_header():
    st.markdown("""
        <div class="header">
            <img src="https://cdn-icons-png.flaticon.com/512/123/123413.png" class="logo">
            <strong>Sistem Prediksi Penjualan Boneka</strong>
        </div>
    """, unsafe_allow_html=True)

# ===== Halaman Dashboard =====
def dashboard_page():
    app_header()
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.subheader("Selamat Datang di Dashboard")
    st.write("Aplikasi ini dirancang untuk membantu memprediksi penjualan boneka menggunakan model Machine Learning (MLP).")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== Halaman Training Model =====
def training_page():
    app_header()
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.subheader("Training Model")

    if st.button("Mulai Training Model"):
        from train_model import train_and_save_model
        train_and_save_model(DATA_PATH, MODEL_PATH)
        st.success("Model berhasil dilatih dan disimpan.")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== Halaman Prediksi =====
def prediksi_page():
    app_header()
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.subheader("Prediksi Penjualan")

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        kategori = st.selectbox("Kategori Produk", ["Boneka", "Bantal"])
        jenis = st.text_input("Jenis Produk")
        harga_satuan = st.number_input("Harga Satuan", min_value=0)
        jumlah_unit = st.number_input("Jumlah Unit", min_value=0)
        diskon = st.slider("Diskon (%)", 0, 100)
        musim_event = st.selectbox("Musim/Event", ["Ramadhan", "Liburan", "Biasa"])
        stok_awal = st.number_input("Stok Awal", min_value=0)
        jenis_transaksi = st.selectbox("Jenis Transaksi", ["Tunai", "Transfer"])
        bulan = st.selectbox("Bulan", list(range(1, 13)))

        fitur = pd.DataFrame([{
            'Kategori': kategori,
            'Jenis': jenis,
            'Harga Satuan': harga_satuan,
            'Jumlah Unit': jumlah_unit,
            'Diskon': diskon,
            'Musim/Event': musim_event,
            'Stok Awal': stok_awal,
            'Jenis Transaksi': jenis_transaksi,
            'Bulan': bulan
        }])

        if st.button("Prediksi"):
            prediksi = model.predict(fitur)
            st.success(f"Prediksi Penjualan: {prediksi[0]:.2f}")
    else:
        st.warning("Model belum dilatih. Silakan latih model terlebih dahulu.")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== Halaman Evaluasi Model =====
def evaluasi_page():
    app_header()
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.subheader("Evaluasi Model")

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)

        y_true = df['Target Penjualan']
        X = df.drop(columns=['Target Penjualan', 'Tanggal'])
        y_pred = model.predict(X)

        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R-squared (R2 Score):** {r2:.2f}")
    else:
        st.warning("Model belum tersedia.")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== Main =====
def main():
    custom_style()
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/123/123413.png", width=60)
    st.sidebar.title("Navigasi")
    menu = st.sidebar.radio("Menu", ["Dashboard", "Training", "Prediksi", "Evaluasi"])

    if menu == "Dashboard":
        dashboard_page()
    elif menu == "Training":
        training_page()
    elif menu == "Prediksi":
        prediksi_page()
    elif menu == "Evaluasi":
        evaluasi_page()

if __name__ == "__main__":
    main()
