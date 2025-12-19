import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Konfigurasi Halaman agar terlihat modern
st.set_page_config(page_title="Butterfly AI Classifier", page_icon="🦋", layout="centered")

# 2. Custom CSS untuk meniru index.html (Dark Mode & Blue Accent)
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    .main-title {
        font-size: 3rem;
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
        padding: 1.5rem;
        border-radius: 1rem;
        margin-top: 20px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #38bdf8, #0ea5e9);
        color: #0f172a !important;
        font-weight: bold;
        border-radius: 0.75rem;
        border: none;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Fungsi Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('kupukupu_model.h5')

model = load_model()

# 4. Header UI
st.markdown('<h1 class="main-title">Butterfly AI Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8;">Computer Vision Project - 75 Species Detection</p>', unsafe_allow_html=True)

# 5. Upload Area
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Layout Kolom untuk hasil
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Gambar Terunggah', use_container_width=True)
    
    with col2:
        st.write("### Analisis Model")
        if st.button('Klasifikasi Sekarang'):
            with st.spinner('Sedang memproses...'):
                # Preprocessing
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prediksi
                predictions = model.predict(img_array)
                score = np.max(predictions)
                
                # Daftar 75 Label milikmu
                labels = ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO', 'ATALA', 'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BECKERS WHITE', 'BLACK HAIRSTREAK', 'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CLEOPATRA', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMMON BANDED AWL', 'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'EASTERN COMA', 'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN', 'ELBOWED PIERROT', 'GOLD BANDED', 'GREAT EGGFLY', 'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREY HAIRSTREAK', 'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'LARGE MARBLE', 'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK', 'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POPINJAY', 'PURPLE HAIRSTREAK', 'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER', 'RED POSTMAN', 'RED SPOTTED PURPLE', 'SCARCE SWALLOW', 'SILVER SPOT SKIPPER', 'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN', 'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES', 'VICEROY', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING']
                
                hasil = labels[np.argmax(predictions)]

                # Tampilan Hasil bergaya Glassmorphism
                st.markdown(f"""
                    <div class="glass-card">
                        <p style="color: #38bdf8; font-size: 0.7rem; font-weight: bold;">SPESIES TERDETEKSI</p>
                        <h2 style="color: #34d399; margin-top: -10px;">{hasil}</h2>
                        <hr style="border-color: rgba(255,255,255,0.1)">
                        <p style="font-size: 0.9rem;">Confidence: <b>{score*100:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.progress(float(score))
else:
    st.info("Silakan unggah foto kupu-kupu untuk memulai deteksi.")

# Footer
st.markdown("<br><p style='text-align: center; color: #475569; font-size: 0.8rem;'>&copy; 2025 Computer Science Project</p>", unsafe_allow_html=True)
