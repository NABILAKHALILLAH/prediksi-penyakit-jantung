import streamlit as st
import pandas as pd
import pickle

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Penyakit Jantung", page_icon="❤️", layout="centered")

# --- Memuat Model ---
# Menggunakan st.cache_resource agar model tidak di-load ulang setiap kali ada perubahan input
@st.cache_resource
def load_model():
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- Tampilan Utama ---
st.title("❤️ Aplikasi Prediksi Penyakit Jantung")
st.write("""
Aplikasi ini memprediksi apakah seseorang memiliki indikasi penyakit jantung berdasarkan 13 parameter medis. 
Silakan masukkan data pasien pada form di bawah ini.
""")

st.divider()

# --- Form Input Data Pasien ---
st.subheader("Data Pasien")

# Membuat layout 2 kolom agar lebih rapi
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur (Age)", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Jenis Kelamin", options=["Perempuan", "Laki-laki"])
    cp = st.selectbox("Tipe Nyeri Dada (Chest Pain Type)", options=[0, 1, 2, 3], help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
    trestbps = st.number_input("Tekanan Darah Rehat (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Kolesterol Serum (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl?", options=["Tidak", "Ya"])
    restecg = st.selectbox("Hasil EKG Rehat (Resting ECG)", options=[0, 1, 2])

with col2:
    thalach = st.number_input("Detak Jantung Maksimal (Thalach)", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Nyeri Dada karena Olahraga? (Exercise Angina)", options=["Tidak", "Ya"])
    oldpeak = st.number_input("Depresi ST karena Olahraga (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Kemiringan Segmen ST (Slope)", options=[0, 1, 2])
    ca = st.selectbox("Jumlah Pembuluh Darah Utama (CA)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (Thal)", options=[0, 1, 2, 3], help="1: Normal, 2: Fixed defect, 3: Reversable defect")

# --- Konversi Input menjadi Format Model ---
# Mengubah input teks menjadi angka (0 atau 1) sesuai format dataset aslinya
sex_val = 1 if sex == "Laki-laki" else 0
fbs_val = 1 if fbs == "Ya" else 0
exang_val = 1 if exang == "Ya" else 0

# Menyusun data ke dalam Pandas DataFrame dengan nama kolom yang SAMA PERSIS dengan dataset
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_val],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs_val],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang_val],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

st.divider()

# --- Tombol Prediksi ---
if st.button("Prediksi Sekarang", type="primary", use_container_width=True):
    # Melakukan prediksi
    prediction = model.predict(input_data)
    
    st.subheader("Hasil Prediksi:")
    
    if prediction[0] == 1:
        st.error("⚠️ **Peringatan:** Model memprediksi bahwa pasien **TERINDIKASI** penyakit jantung. Segera konsultasikan ke dokter.")
    else:
        st.success("✅ **Aman:** Model memprediksi bahwa pasien **TIDAK TERINDIKASI** penyakit jantung.")

st.caption("Catatan: Ini adalah prediksi Machine Learning berbasis Logistic Regression dan tidak menggantikan diagnosis medis profesional.")
