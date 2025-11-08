import uvicorn
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import io
import sqlite3
from datetime import datetime

# 1. Inisialisasi Aplikasi FastAPI 
app = FastAPI(title="API Deteksi Kesehatan Terumbu Karang")

MODEL_PATH = "coral_model_best.h5"
IMG_HEIGHT = 224
IMG_WIDTH = 224
class_names = ['Bleached', 'Healthy'] 

DB_NAME = "predictions.db"

# 2. Load Model 
try:
    model = load_model(MODEL_PATH)
    print(f"Model {MODEL_PATH} berhasil di-load.")
except Exception as e:
    print(f"ERROR: Gagal me-load model. {e}")
    model = None

# 3. Database Helper 
def init_db():
    """Inisialisasi database SQLite"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        filename TEXT,
        predicted_label TEXT,
        confidence REAL,
        raw_probability REAL
    )
    ''')
    conn.commit()
    conn.close()

def log_prediction(filename, label, confidence, raw_proba):
    """Mencatat hasil prediksi ke database"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (filename, predicted_label, confidence, raw_probability) VALUES (?, ?, ?, ?)",
        (filename, label, confidence, raw_proba)
    )
    conn.commit()
    conn.close()

# Inisialisasi DB saat startup
init_db()

# 4. Fungsi Helper untuk Preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Melakukan preprocessing pada gambar (SAMA PERSIS seperti di notebook)"""
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image_array = img_to_array(image)
    image_array = image_array / 255.0  
    image_array = np.expand_dims(image_array, axis=0) 
    return image_array

# 5. Endpoint Prediksi 
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint utama untuk prediksi gambar.
    """
    if not model:
        return {"error": "Model tidak ter-load."}

    # 1. Baca dan preprocess gambar
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed_image = preprocess_image(image)
    
    # 2. Lakukan prediksi
    prediction_proba = model.predict(processed_image)[0][0]
    
    # 3. Tentukan kelas 
    if prediction_proba > 0.5:
        class_index = 1
        confidence = prediction_proba
    else:
        class_index = 0
        confidence = 1 - prediction_proba
        
    class_label = class_names[class_index]
    
    # 4. Log ke Database 
    log_prediction(
        file.filename, 
        class_label, 
        float(confidence), 
        float(prediction_proba)
    )
    
    # 5. Kembalikan hasil sebagai JSON
    return {
        "filename": file.filename,
        "predicted_label": class_label,
        "confidence_score": float(confidence),
        "raw_probability": float(prediction_proba)
    }

# 6. Jalankan Server (Hanya untuk testing lokal) 
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)