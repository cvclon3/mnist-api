# Импорт необходимых библиотек
import numpy as np  # Для работы с числовыми массивами
import logging  # Для логирования событий
from fastapi import FastAPI, UploadFile, File, HTTPException  # Основные компоненты FastAPI
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse  # Типы ответов API
from fastapi.middleware.cors import CORSMiddleware  # Для обработки CORS
from fastapi.staticfiles import StaticFiles  # Для обслуживания статических файлов
from PIL import Image  # Для работы с изображениями
import io  # Для работы с потоками данных
import tensorflow as tf  # Для работы с ML моделью
from pathlib import Path  # Для работы с путями файлов
import time  # Для измерения времени выполнения
import os  # Для работы с файловой системой

# Настройка системы логирования
logging.basicConfig(level=logging.INFO)  # Устанавливаем уровень логирования INFO
logger = logging.getLogger(__name__)  # Создаем логгер для текущего модуля

# Создание экземпляра FastAPI приложения с метаданными
app = FastAPI(
    title="MNIST Digit Recognition API",  # Название API
    description="API для распознавания рукописных цифр с использованием модели Keras",  # Описание
    version="1.0.0"  # Версия API
)

# Настройка CORS (Cross-Origin Resource Sharing)
# Это необходимо для доступа к API из браузерных приложений на других доменах
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем запросы со всех доменов
    allow_methods=["*"],  # Разрешаем все HTTP методы
    allow_headers=["*"],  # Разрешаем все заголовки
)

# Создаем директорию для тестовых изображений, если её нет
os.makedirs("static/test_digits", exist_ok=True)
# Монтируем статическую директорию для обслуживания файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# Загрузка ML модели
MODEL_PATH = 'my_mnist_model_savedmodel.keras'  # Путь к файлу модели
try:
    # Загружаем модель с помощью TensorFlow/Keras
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Модель успешно загружена из {MODEL_PATH}")
except Exception as e:
    # В случае ошибки загрузки модели логируем и прекращаем работу
    logger.error(f"Ошибка загрузки модели: {str(e)}")
    raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Предварительная обработка изображения для подачи в модель.

    Параметры:
        image: PIL.Image - исходное изображение

    Возвращает:
        np.ndarray - обработанный массив numpy, готовый для модели
    """
    # Конвертируем в черно-белый формат и изменяем размер до 28x28 (как в MNIST)
    image = image.convert('L').resize((28, 28))
    # Преобразуем изображение в numpy массив
    image_array = np.array(image)
    # Инвертируем цвета (MNIST использует белые цифры на черном фоне)
    image_array = 255 - image_array
    # Нормализуем значения пикселей к диапазону [0, 1]
    image_array = image_array.astype('float32') / 255.0
    # Изменяем форму массива для соответствия ожиданиям модели:
    # (batch_size, height, width, channels) - здесь batch_size=1, channels=1
    return image_array.reshape(1, 28, 28, 1)


@app.get("/", response_class=HTMLResponse)
async def serve_interface():
    """
    Основной endpoint, возвращающий HTML интерфейс.

    Возвращает:
        FileResponse - HTML страницу из статической директории
    """
    return FileResponse("static/index.html")


@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    """
    Endpoint для распознавания цифры на изображении.

    Параметры:
        file: UploadFile - файл изображения, загружаемый пользователем

    Возвращает:
        JSONResponse - результат распознавания в формате JSON:
            - digit: распознанная цифра (0-9)
            - confidence: уверенность модели (0-1)
            - processing_time: время обработки в секундах

    Исключения:
        HTTPException 400: если загружен не файл изображения
        HTTPException 500: при ошибках обработки изображения
    """
    start_time = time.time()  # Засекаем время начала обработки

    # Проверяем, что загруженный файл является изображением
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="Требуется изображение")

    try:
        # Читаем содержимое загруженного файла
        contents = await file.read()
        # Создаем объект изображения из байтов
        image = Image.open(io.BytesIO(contents))
        # Обрабатываем изображение для модели
        image_array = preprocess_image(image)

        # Получаем предсказание модели
        prediction = model.predict(image_array)
        # Определяем цифру с максимальной вероятностью
        digit = int(np.argmax(prediction))
        # Получаем значение уверенности модели
        confidence = float(np.max(prediction))

        # Возвращаем результат
        return {
            "digit": digit,
            "confidence": confidence,
            "processing_time": time.time() - start_time  # Вычисляем время выполнения
        }
    except Exception as e:
        # Логируем ошибку и возвращаем 500 статус
        logger.error(f"Ошибка: {str(e)}", exc_info=True)
        raise HTTPException(500, detail="Ошибка обработки изображения")


@app.get("/examples/")
async def get_examples_list():
    """
    Endpoint для получения списка доступных тестовых изображений.

    Возвращает:
        JSONResponse - список имен файлов примеров без расширений
    """
    # Получаем список всех .jpg файлов в директории test_digits
    examples = [f.stem for f in Path("static/test_digits").glob("*.jpg")]
    return {"examples": examples}


# Точка входа при запуске файла напрямую
if __name__ == "__main__":
    import uvicorn

    # Запускаем сервер Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)