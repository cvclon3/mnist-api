import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import tensorflow as tf
from pathlib import Path
import time
import os

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MNIST Digit Recognition API",
    description="API для распознавания рукописных цифр с использованием модели Keras",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование статики
os.makedirs("static/test_digits", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Загрузка модели
MODEL_PATH = 'my_mnist_model_savedmodel.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Модель успешно загружена из {MODEL_PATH}")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {str(e)}")
    raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Предобработка изображения для модели MNIST"""
    image = image.convert('L').resize((28, 28))
    image_array = 255 - np.array(image)  # Инверсия цветов
    image_array = image_array.astype('float32') / 255.0
    return image_array.reshape(1, 28, 28, 1)


@app.get("/", response_class=HTMLResponse)
async def serve_interface():
    """Возвращает HTML-интерфейс"""
    return FileResponse("static/index.html")


@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    """Распознавание рукописной цифры"""
    start_time = time.time()

    if not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="Требуется изображение")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = preprocess_image(image)

        prediction = model.predict(image_array)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "digit": digit,
            "confidence": confidence,
            "processing_time": time.time() - start_time
        }
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        raise HTTPException(500, detail="Ошибка обработки изображения")


@app.get("/examples/")
async def get_examples_list():
    """Возвращает список доступных примеров"""
    examples = [f.stem for f in Path("static/test_digits").glob("*.jpg")]
    return {"examples": examples}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)