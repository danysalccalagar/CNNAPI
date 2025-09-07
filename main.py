from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import io
from PIL import Image
import os

app = FastAPI()

# Cargar modelo (ajusta la ruta a tu archivo .h5 o .keras)
MODEL_PATH = "modelo/clasificador_gorgojo.h5"
model = load_model(MODEL_PATH)

# Tamaño de entrada esperado por el modelo
IMG_SIZE = (128, 128)   # 🔹 Tu modelo espera 128x128x3

def preprocess(img: Image.Image):
    img = img.resize(IMG_SIZE)             # Redimensionar
    img_array = image.img_to_array(img)    # Convertir a array
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizar y añadir batch
    return img_array

@app.get("/")
async def root():
    return {"status": "ok", "message": "API funcionando ✅"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    input_data = preprocess(img)
    prediction = model.predict(input_data)
    result = int((prediction > 0.5).astype("int32")[0][0])
    return {"resultado": result, "confianza": float(prediction[0][0])}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render asigna el puerto
    uvicorn.run(app, host="0.0.0.0", port=port)
