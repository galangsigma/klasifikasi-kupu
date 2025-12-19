import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# 1. Konfigurasi Halaman & Tema
st.set_page_config(page_title="Butterfly AI Classifier", page_icon="🦋", layout="wide")

# 2. Suntikan CSS Kustom (Diperkaya untuk mengisi ruang)
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #38bdf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* Kartu Statistik di Sidebar */
    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 1.5rem;
        border-left: 5px solid #38bdf8;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    /* Membuat area upload lebih menarik */
    .stFileUploader {
        border: 2px dashed #38bdf8;
        border-radius: 15px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar (Mengisi ruang kosong di kiri)
with st.sidebar:
    st.markdown("## 📊 Info Model")
    st.markdown("""
    <div class="stat-card">
        <small>Arsitektur</small><br>
        <b>MobileNetV2</b>
    </div>
    <div class="stat-card">
        <small>Jumlah Kelas</small><br>
        <b>75 Spesies</b>
    </div>
    <div class="stat-card">
        <small>Input Size</small><br>
        <b>224x224 px</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("## 🛠️ Cara Kerja")
    st.info("Model ini mengekstraksi fitur pola sayap dan warna untuk menentukan spesies dengan probabilitas tertinggi.")

# 4. Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('kupukupu_model.h5')

model = load_model()
labels = ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO', 'ATALA', 'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BECKERS WHITE', 'BLACK HAIRSTREAK', 'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CLEOPATRA', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMMON BANDED AWL', 'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'EASTERN COMA', 'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN', 'ELBOWED PIERROT', 'GOLD BANDED', 'GREAT EGGFLY', 'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREY HAIRSTREAK', 'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'LARGE MARBLE', 'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK', 'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POPINJAY', 'PURPLE HAIRSTREAK', 'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER', 'RED POSTMAN', 'RED SPOTTED PURPLE', 'SCARCE SWALLOW', 'SILVER SPOT SKIPPER', 'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN', 'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES', 'VICEROY', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING']

# 5. Antarmuka Utama (Layout Kolom)
st.markdown('<h1 class="main-title">Butterfly AI Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 1.2rem;">Sistem Identifikasi Spesies Otomatis Berbasis Deep Learning</p>', unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Input Gambar")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Preview Gambar', use_container_width=True)
    else:
        st.markdown("""
            <div style="background: rgba(56, 189, 248, 0.1); padding: 40px; border-radius: 15px; text-align: center; border: 1px dashed #38bdf8;">
                <h1 style="font-size: 50px;">🦋</h1>
                <p style="color: #38bdf8;">Silakan unggah foto untuk memulai</p>
            </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🤖 Hasil Identifikasi")
    if uploaded_file is not None:
        if st.button('Mulai Analisis Sistem'):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            # Preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            predictions = model.predict(img_array)
            score = np.max(predictions)
            hasil = labels[np.argmax(predictions)]

            # Tampilan Hasil
            st.markdown(f"""
                <div class="glass-card">
                    <p style="color: #38bdf8; font-size: 0.8rem; font-weight: bold; letter-spacing: 2px;">PREDIKSI TERBAIK</p>
                    <h1 style="color: white; margin-top: -10px; font-size: 2.5rem;">{hasil}</h1>
                    <div style="background: rgba(52, 211, 153, 0.1); padding: 10px; border-radius: 8px; border: 1px solid #34d399;">
                        <span style="color: #34d399; font-weight: bold;">Tingkat Keyakinan: {score*100:.2f}%</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.8rem; mt-3">Identifikasi selesai menggunakan analisis neural network.</p>
                </div>
                """, unsafe_allow_html=True)
            
            if score < 0.5:
                st.warning("⚠️ Skor rendah. Pastikan objek kupu-kupu terlihat jelas dan dominan.")
    else:
        st.write("Menunggu input gambar...")

# 6. Footer (Tetap rapi)
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<hr><center style='color: #475569;'>&copy; 2025 Computer Science Project | Binus University</center>", unsafe_allow_html=True)
