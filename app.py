import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Klasifikasi Spesies Kupu-Kupu 🦋")

# Load model (pastikan nama file sesuai)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('kupukupu_model.h5')

model = load_model()

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar kupu-kupu...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Preprocessing (sesuaikan dengan ukuran input model kamu, misal 224x224)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    if st.button('Klasifikasi'):
        predictions = model.predict(img_array)
        # Ganti labels sesuai dengan urutan kelas di model kamu
        labels = ['Danaus Plexippus', 'Heliconius Charithonia', 'Limenitis Arthemis'] 
        hasil = labels[np.argmax(predictions)]

        st.success(f"Hasil Prediksi: **{hasil}**")

