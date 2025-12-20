import streamlit as st
import tensorflow as tf
import tensorflow.keras.preprocessing.image as image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
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
labels = [
    "Adonis Butterfly",
    "African Giant Swallowtail Butterfly",
    "American Snoot Butterfly",
    "Anna's Eighty-Eight Butterfly",
    "Appollo Butterfly",
    "Atala Butterfly",
    "Banded Orange Heliconian Butterfly",
    "Banded Peacock Butterfly",
    "Beckers White Butterfly",
    "Black Hairstreak Butterfly",
    "Blue Morpho Butterfly",
    "Blue Spotted Crow Butterfly",
    "Brown Siproeta Butterfly",
    "Cabbage White Butterfly",
    "Cairns Birdwing Butterfly",
    "Checquered Skipper Butterfly",
    "Chestnut Butterfly",
    "Cleopatra Butterfly",
    "Clodius Parnassian Butterfly",
    "Clouded Sulphur Butterfly",
    "Common Banded Awl Butterfly",
    "Common Wood-Nymph Butterfly",
    "Copper Tail Butterfly",
    "Crecent Butterfly",
    "Crimson Patch Butterfly",
    "Danaid Eggfly Butterfly",
    "Eastern Coma Butterfly",
    "Eastern Dapple White Butterfly",
    "Eastern Pine Elfin Butterfly",
    "Elbowed Pierrot Butterfly",
    "Gold Banded Butterfly",
    "Great Eggfly Butterfly",
    "Great Jay Butterfly",
    "Green Celled Cattleheart Butterfly",
    "Grey Hairstreak Butterfly",
    "Indra Swallow Butterfly",
    "Iphiclus Sister Butterfly",
    "Julia Butterfly",
    "Large Marble Butterfly",
    "Malachite Butterfly",
    "Mangrove Skipper Butterfly",
    "Mestra Butterfly",
    "Metalmark Butterfly",
    "Milberts Tortoiseshell Butterfly",
    "Monarch Butterfly",
    "Mourning Cloak Butterfly",
    "Orange Oakleaf Butterfly",
    "Orange Tip Butterfly",
    "Orchard Swallow Butterfly",
    "Painted Lady Butterfly",
    "Paper Kite Butterfly",
    "Peacock Butterfly",
    "Pine White Butterfly",
    "Pipevine Swallow Butterfly",
    "Popinjay Butterfly",
    "Purple Hairstreak Butterfly",
    "Purplish Copper Butterfly",
    "Question Mark Butterfly",
    "Red Admiral Butterfly",
    "Red Cracker Butterfly",
    "Red Postman Butterfly",
    "Red Spotted Purple Butterfly",
    "Scarce Swallow Butterfly",
    "Silver Spot Skipper Butterfly",
    "Sleepy Orange Butterfly",
    "Sootywing Butterfly",
    "Southern Dogface Butterfly",
    "Straited Queen Butterfly",
    "Tropical Leafwing Butterfly",
    "Two Barred Flasher Butterfly",
    "Ulyses Butterfly",
    "Viceroy Butterfly",
    "Wood Satyr Butterfly",
    "Yellow Swallow Tail Butterfly",
    "Zebra Long Wing Butterfly",
]

# 4. Header Utama
st.markdown('<h1 class="main-title">Butterfly AI Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 1.1rem; margin-top: 10px;">Sistem Klasifikasi Spesies Kupu-Kupu</p>', unsafe_allow_html=True)
st.write("---")

# 5. Konten Tengah
# Area Upload
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # Tampilkan preview gambar di tengah
    st.image(img, caption='Gambar yang diunggah', use_container_width=True)
    
    if st.button('MULAI IDENTIFIKASI'):
        with st.spinner('Menganalisis gambar...'):
            # Preprocessing
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            # Prediksi
            predictions = model.predict(img)
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
