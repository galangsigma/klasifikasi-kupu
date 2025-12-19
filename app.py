import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# 1. Konfigurasi Halaman (Centered agar fokus di tengah)
st.set_page_config(page_title="Butterfly AI Classifier", page_icon="🦋", layout="centered")

# 2. Custom CSS untuk Tampilan Rapi & Modern (Tanpa Sidebar)
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
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

    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        border-radius: 1.5rem;
        margin-top: 20px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #38bdf8, #0ea5e9);
        color: #0f172a !important;
        font-weight: bold;
        border-radius: 0.75rem;
        border: none;
        padding: 0.7rem;
        font-size: 1.1rem;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.4);
    }

    .upload-box {
        border: 2px dashed #38bdf8;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        background: rgba(56, 189, 248, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Fungsi Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('kupukupu_model.h5')

model = load_model()
labels = ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO', 'ATALA', 'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BECKERS WHITE', 'BLACK HAIRSTREAK', 'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CLEOPATRA', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMMON BANDED AWL', 'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'EASTERN COMA', 'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN', 'ELBOWED PIERROT', 'GOLD BANDED', 'GREAT EGGFLY', 'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREY HAIRSTREAK', 'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'LARGE MARBLE', 'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK', 'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POPINJAY', 'PURPLE HAIRSTREAK', 'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER', 'RED POSTMAN', 'RED SPOTTED PURPLE', 'SCARCE SWALLOW', 'SILVER SPOT SKIPPER', 'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN', 'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES', 'VICEROY', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING']

# 4. Header Utama
st.markdown('<h1 class="main-title">Butterfly AI Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 1.1rem; margin-top: 10px;">Sistem Klasifikasi Spesies Kupu-Kupu</p>', unsafe_allow_html=True)
st.write("---")

# 5. Konten Tengah
# Area Upload
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Tampilkan preview gambar di tengah
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)
    
    if st.button('MULAI IDENTIFIKASI'):
        with st.spinner('Menganalisis gambar...'):
            # Preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            predictions = model.predict(img_array)
            score = np.max(predictions)
            
            # Logika Skor > 55%
            if score >= 0.55:
                hasil = labels[np.argmax(predictions)]
                
                st.markdown(f"""
                    <div class="glass-card">
                        <p style="color: #38bdf8; font-size: 0.9rem; font-weight: bold; letter-spacing: 2px;">HASIL IDENTIFIKASI</p>
                        <h1 style="color: white; margin-top: -10px; font-size: 2.8rem;">{hasil}</h1>
                        <div style="background: rgba(52, 211, 153, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #34d399; margin-top: 15px;">
                            <span style="color: #34d399; font-weight: bold; font-size: 1.1rem;">Akurasi: {score*100:.2f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Jika skor di bawah 55%
                st.error("⚠️ Maaf, sistem tidak dapat mengidentifikasi spesies dengan cukup yakin. Pastikan gambar kupu-kupu terlihat jelas.")
                st.info(f"Keyakinan model hanya {(score*100):.2f}%, di bawah batas minimum 55%.")

else:
    st.markdown("""
        <div class="upload-box">
            <h1 style="font-size: 50px; margin-bottom: 0;">🦋</h1>
            <p style="color: #38bdf8; font-weight: 500;">Silakan pilih atau tarik gambar ke sini</p>
        </div>
    """, unsafe_allow_html=True)

# 6. Footer
st.markdown("<br><br><hr><center style='color: #475569;'>&copy; 2025 Butterfly AI Project</center>", unsafe_allow_html=True)
