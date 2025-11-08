import streamlit as st
import requests
from PIL import Image
import io

# Konfigurasi Halaman 
st.set_page_config(
    page_title="Deteksi Kesehatan Karang",
    page_icon="🪸",
    layout="centered"
)

# URL Backend API 
API_URL = "http://127.0.0.1:8000/predict"

# Judul & Deskripsi
st.title("🪸 Sistem Deteksi Kesehatan Terumbu Karang")
st.write("""
Unggah gambar terumbu karang untuk diprediksi. 
Model akan mengklasifikasikan apakah karang tersebut 'Healthy' (Sehat) atau 'Bleached' (Memutih).
""")

# File Uploader 
uploaded_file = st.file_uploader(
    "Pilih sebuah gambar...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 1. Tampilkan gambar yang di-upload
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    # 2. Buat tombol untuk prediksi
    if st.button("Analisis Gambar Ini"):
        
        # Tampilkan spinner selagi menunggu
        with st.spinner("Model sedang menganalisis..."):
            
            # 3. Siapkan file untuk dikirim ke API
            file_bytes = io.BytesIO(uploaded_file.getvalue())
            files_to_send = {'file': (uploaded_file.name, file_bytes, uploaded_file.type)}
            
            try:
                # 4. Kirim request ke API Backend
                response = requests.post(API_URL, files=files_to_send)
                
                if response.status_code == 200:
                    # 5. Jika sukses, tampilkan hasil
                    data = response.json()
                    label = data['predicted_label']
                    confidence = data['confidence_score'] * 100
                    
                    if label == 'Healthy':
                        st.success(f"**Prediksi: {label}** (Keyakinan: {confidence:.2f}%)")
                    else:
                        st.error(f"**Prediksi: {label}** (Keyakinan: {confidence:.2f}%)")
                        
                    st.write(f"Probabilitas mentah: {data['raw_probability']:.4f}")
                    
                else:
                    # Jika API mengembalikan error
                    st.error(f"Error dari API: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                # Jika backend tidak terhubung (misal, server API mati)
                st.error(f"Gagal terhubung ke API: {e}")