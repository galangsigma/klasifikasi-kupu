import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Konfigurasi Halaman & Tema
st.set_page_config(page_title="Butterfly AI Classifier", page_icon="🦋", layout="centered")

# 2. Suntikan CSS Kustom (Meniru gaya Tailwind di file index.html kamu)
st.markdown("""
    <style>
    /* Mengubah latar belakang utama */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Header Style */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #38bdf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Kartu Glassmorphism */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 1.5rem;
        border-left: 5px solid #34d399;
    }
    
    /* Tombol Kustom */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #38bdf8, #0ea5e9);
        color: #0f172a;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 0.75rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('kupukupu_model.h5')

model = load_model()
# Pastikan urutan label ini sama dengan urutan folder saat training
labels = ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO', 'ATALA', 'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BECKERS WHITE', 'BLACK HAIRSTREAK', 'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CLEOPATRA', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMMON BANDED AWL', 'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'EASTERN COMA', 'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN', 'ELBOWED PIERROT', 'GOLD BANDED', 'GREAT EGGFLY', 'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREY HAIRSTREAK', 'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'LARGE MARBLE', 'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK', 'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POPINJAY', 'PURPLE HAIRSTREAK', 'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER', 'RED POSTMAN', 'RED SPOTTED PURPLE', 'SCARCE SWALLOW', 'SILVER SPOT SKIPPER', 'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN', 'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES', 'VICEROY', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING']

# 4. Antarmuka Utama
st.markdown('<h1 class="main-title">Butterfly AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8;">Implementasi Computer Vision menggunakan Deep Learning</p>', unsafe_allow_html=True)
st.write("---")

# Area Upload
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Tampilkan Gambar dengan Frame Bagus
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)
    
    if st.button('IDENTIFIKASI SEKARANG'):
        with st.spinner('Menganalisis pixel...'):
            # Preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            predictions = model.predict(img_array)
            score = np.max(predictions)
            hasil = labels[np.argmax(predictions)]

            # 5. Tampilan Hasil Prediksi (Gaya Kartu Sukses)
            st.markdown(f"""
                <div class="glass-card">
                    <p style="color: #38bdf8; font-size: 0.8rem; font-weight: bold; letter-spacing: 0.1em;">HASIL PREDIKSI</p>
                    <h2 style="color: white; margin-top: -10px;">{hasil}</h2>
                    <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                        <span style="color: #94a3b8;">Tingkat Keyakinan</span>
                        <span style="color: #34d399; font-weight: bold;">{score*100:.2f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Progress bar untuk visualisasi akurasi
            st.progress(float(score))

else:
    # Tampilan instruksi jika belum ada gambar
    st.info("💡 Tips: Gunakan gambar dengan pencahayaan terang untuk hasil lebih akurat.")

# Footer
st.markdown("<br><hr><center style='color: #64748b;'>&copy; 2025 Computer Science Project</center>", unsafe_allow_html=True)
