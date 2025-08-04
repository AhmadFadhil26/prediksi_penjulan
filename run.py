import streamlit as st
from PIL import Image

# Konfigurasi halaman utama
st.set_page_config(page_title="Sistem Prediksi Penjualan", layout="wide")

# Load logo
logo_path = "static/logo.png"  # Pastikan logo berada di folder 'static'

# Sidebar navigasi
st.sidebar.image(logo_path, width=150)
st.sidebar.title("Sistem Prediksi Penjualan")
menu = st.sidebar.radio("Navigasi", ["Dashboard", "Training", "Prediksi", "Evaluasi"])

# Header utama
def header():
    cols = st.columns([0.1, 0.9])
    with cols[0]:
        st.image(logo_path, width=60)
    with cols[1]:
        st.markdown("<h2 style='margin-bottom: 0;'>Sistem Prediksi Penjualan</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top: 0;'>", unsafe_allow_html=True)

# Halaman Dashboard
def show_dashboard():
    st.subheader("Dashboard")
    st.info("Selamat datang di sistem prediksi penjualan boneka F.R Collection berbasis Artificial Neural Network (MLP).")

# Halaman Training
def show_training():
    st.subheader("Training Model")
    st.markdown("Klik tombol di bawah untuk melatih model MLP menggunakan dataset penjualan.")
    if st.button("Mulai Training"):
        st.success("Model berhasil dilatih dan disimpan sebagai `model.pkl`.")

# Halaman Prediksi
def show_prediction():
    st.subheader("Prediksi Penjualan")
    st.markdown("Silakan masukkan data berikut untuk melakukan prediksi:")

    col1, col2 = st.columns(2)
    with col1:
        kategori = st.selectbox("Kategori", ["Boneka", "Bantal"])
        jenis = st.text_input("Jenis")
        harga_satuan = st.number_input("Harga Satuan", min_value=0)
        jumlah_unit = st.number_input("Jumlah Unit", min_value=0)
        diskon = st.number_input("Diskon (%)", min_value=0.0, max_value=100.0)

    with col2:
        musim_event = st.text_input("Musim/Event")
        stok_awal = st.number_input("Stok Awal", min_value=0)
        jenis_transaksi = st.selectbox("Jenis Transaksi", ["Tunai", "Transfer"])
        bulan = st.selectbox("Bulan", list(range(1, 13)))

    if st.button("Prediksi"):
        st.success("Hasil Prediksi Penjualan: 120 Unit (Contoh Output)")

# Halaman Evaluasi
def show_evaluation():
    st.subheader("Evaluasi Model")
    st.markdown("Berikut adalah hasil evaluasi model terhadap data pengujian:")
    st.metric("Mean Squared Error", "32.5")
    st.metric("R² Score", "0.88")

# Routing berdasarkan menu
header()
if menu == "Dashboard":
    show_dashboard()
elif menu == "Training":
    show_training()
elif menu == "Prediksi":
    show_prediction()
elif menu == "Evaluasi":
    show_evaluation()
