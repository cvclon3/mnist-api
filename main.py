import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tensorflow as tf
from pathlib import Path
import time

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MNIST Digit Recognition API",
    description="API для распознавания рукописных цифр с использованием модели Keras",
    version="1.0.0"
)

# Настройка CORS (для доступа из браузера)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели
MODEL_PATH = 'my_mnist_model_savedmodel.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Модель успешно загружена из {MODEL_PATH}")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {str(e)}")
    raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.post("/predict/", response_model=dict)
async def predict_digit(file: UploadFile = File(...)):
    """
    Распознавание рукописной цифры на изображении

    Параметры:
    - file: Изображение в формате PNG/JPG/JPEG

    Возвращает:
    - digit: Распознанная цифра (0-9)
    - confidence: Уверенность модели (0-1)
    - processing_time: Время обработки в секундах
    """
    start_time = time.time()

    # Проверка типа файла
    if not file.content_type.startswith('image/'):
        error_msg = f"Неподдерживаемый тип файла: {file.content_type}"
        logger.warning(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        # Чтение и предобработка изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')

        # Логгирование информации об изображении
        logger.info(f"Получено изображение: {file.filename}, размер: {image.size}, режим: {image.mode}")

        # Подготовка изображения для модели
        image = image.resize((28, 28))
        image_array = 255 - np.array(image)  # Инверсия цветов как в MNIST
        image_array = image_array.astype('float32') / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        # Предсказание
        prediction = model.predict(image_array)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        processing_time = time.time() - start_time

        logger.info(f"Предсказание: цифра {predicted_digit} с уверенностью {confidence:.2f}")

        return {
            "digit": predicted_digit,
            "confidence": confidence,
            "processing_time": processing_time
        }

    except Exception as e:
        error_msg = f"Ошибка обработки изображения: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


# Пример тестового запроса для документации
@app.get("/test/", include_in_schema=False)
async def test_prediction():
    """Тестовый запрос с примером изображения"""
    test_image_path = Path("static/test_digit.png")
    if not test_image_path.exists():
        raise HTTPException(status_code=404, detail="Тестовое изображение не найдено")

    with open(test_image_path, "rb") as f:
        return await predict_digit(UploadFile(file=f, filename="test_digit.png"))