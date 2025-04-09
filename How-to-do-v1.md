# Полное методическое руководство по разработке API для распознавания рукописных цифр

## Введение

В этом подробном руководстве я расскажу о полном процессе разработки системы распознавания рукописных цифр с использованием современных технологий. Проект включает в себя:

1. Серверную часть на FastAPI
2. Модель машинного обучения на TensorFlow/Keras
3. Интерактивный веб-интерфейс
4. API для интеграции с другими системами

## Содержание

1. [Технологический стек и инструменты](#технологический-стек-и-инструменты)
2. [Архитектура системы](#архитектура-системы)
3. [Подготовка окружения](#подготовка-окружения)
4. [Разработка серверной части](#разработка-серверной-части)
   - [Инициализация FastAPI](#инициализация-fastapi)
   - [Работа с моделью](#работа-с-моделью)
   - [API endpoints](#api-endpoints)
5. [Клиентская часть](#клиентская-часть)
   - [Структура интерфейса](#структура-интерфейса)
   - [Логика работы](#логика-работы)
6. [Взаимодействие компонентов](#взаимодействие-компонентов)
7. [Развертывание](#развертывание)
8. [Тестирование](#тестирование)
9. [Возможные проблемы](#возможные-проблемы)
10. [Дальнейшее развитие](#дальнейшее-развитие)

## Технологический стек и инструменты

### Основные технологии:
- **Python 3.9+** - основной язык серверной части
- **FastAPI** - современный фреймворк для создания API
- **TensorFlow 2.x** - фреймворк машинного обучения
- **Keras** - высокоуровневый API для нейронных сетей
- **Pillow (PIL)** - обработка изображений
- **NumPy** - работа с числовыми массивами
- **Uvicorn** - ASGI-сервер для запуска приложения

### Клиентские технологии:
- **HTML5** - структура веб-страницы
- **CSS3** - стилизация интерфейса
- **JavaScript** - интерактивность и работа с API
- **Canvas API** - реализация рисования цифр

### Вспомогательные инструменты:
- **Visual Studio Code** - среда разработки
- **Postman** - тестирование API
- **Docker** - контейнеризация приложения
- **Git** - контроль версий

## Архитектура системы

Система состоит из трех основных компонентов:

1. **Серверное приложение** (main.py):
   - FastAPI приложение
   - Модель машинного обучения
   - API endpoints

2. **Клиентский интерфейс** (index.html):
   - Canvas для рисования цифр
   - Форма загрузки изображений
   - Отображение результатов

3. **Дополнительные ресурсы**:
   - Сохраненная модель Keras
   - Тестовые изображения цифр

```
mnist-recognition-api/
├── main.py                # Основной серверный файл
├── my_mnist_model.keras   # Обученная модель
├── requirements.txt       # Зависимости Python
└── static/
    ├── index.html         # Клиентский интерфейс
    ├── styles.css         # Стили (опционально)
    └── test_digits/       # Тестовые изображения
        ├── 0.jpg          # Пример цифры 0
        ├── 1.jpg          # Пример цифры 1
        ...
        └── 9.jpg          # Пример цифры 9
```

## Подготовка окружения

### 1. Установка Python

Рекомендуется использовать Python 3.9 или новее. Скачать можно с [официального сайта](https://www.python.org/downloads/).

### 2. Создание виртуального окружения

```bash
python -m venv venv
```

Активация окружения:

- Для Windows:
  ```bash
  venv\Scripts\activate
  ```

- Для Linux/MacOS:
  ```bash
  source venv/bin/activate
  ```

### 3. Установка зависимостей

Создайте файл `requirements.txt` со следующим содержимым:

```text
fastapi==0.109.1
uvicorn==0.27.0
numpy==1.26.4
Pillow==10.2.0
tensorflow==2.15.0
python-multipart==0.0.9
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

## Разработка серверной части

### Инициализация FastAPI

Основной файл сервера `main.py` начинается с импорта необходимых модулей:

```python
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
```

Далее создаем экземпляр FastAPI приложения с метаданными:

```python
app = FastAPI(
    title="MNIST Digit Recognition API",
    description="API для распознавания рукописных цифр с использованием модели Keras",
    version="1.0.0"
)
```

### Настройка CORS

Для разрешения запросов из браузерных приложений на других доменах:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем запросы со всех доменов
    allow_methods=["*"],  # Разрешаем все HTTP методы
    allow_headers=["*"],  # Разрешаем все заголовки
)
```

### Работа с моделью

Загрузка предварительно обученной модели:

```python
MODEL_PATH = 'my_mnist_model.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Модель успешно загружена из {MODEL_PATH}")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {str(e)}")
    raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")
```

Функция предобработки изображений:

```python
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Подготовка изображения для модели MNIST"""
    # Конвертируем в черно-белый и изменяем размер
    image = image.convert('L').resize((28, 28))
    # Преобразуем в numpy массив
    image_array = np.array(image)
    # Инвертируем цвета (MNIST - белые цифры на черном фоне)
    image_array = 255 - image_array
    # Нормализуем значения пикселей
    image_array = image_array.astype('float32') / 255.0
    # Изменяем форму для модели (batch_size=1, height=28, width=28, channels=1)
    return image_array.reshape(1, 28, 28, 1)
```

### API Endpoints

Основной endpoint для распознавания:

```python
@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    """Распознавание рукописной цифры на изображении"""
    start_time = time.time()
    
    # Проверка типа файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="Требуется изображение")
    
    try:
        # Чтение и обработка изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = preprocess_image(image)
        
        # Получение предсказания
        prediction = model.predict(image_array)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        return {
            "digit": digit,
            "confidence": confidence,
            "processing_time": time.time() - start_time
        }
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}", exc_info=True)
        raise HTTPException(500, detail="Ошибка обработки изображения")
```

Endpoint для получения списка тестовых изображений:

```python
@app.get("/examples/")
async def get_examples_list():
    """Возвращает список доступных тестовых изображений"""
    examples = [f.stem for f in Path("static/test_digits").glob("*.jpg")]
    return {"examples": examples}
```

Endpoint для главной страницы:

```python
@app.get("/", response_class=HTMLResponse)
async def serve_interface():
    """Возвращает HTML интерфейс"""
    return FileResponse("static/index.html")
```

## Клиентская часть

### Структура интерфейса

Файл `static/index.html` содержит:

1. **Canvas** для рисования цифр
2. **Кнопки управления**:
   - "Очистить" - очистка canvas
   - "Пример" - загрузка тестового изображения
   - "Распознать" - отправка изображения на сервер
3. **Форма загрузки** изображений
4. **Блок результатов** с отображением:
   - Распознанной цифры
   - Уверенности модели
   - Времени обработки

### Логика работы

Основные JavaScript функции:

1. **Обработка рисования** на canvas:

```javascript
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

// Настройки рисования
ctx.strokeStyle = 'white';
ctx.lineWidth = 15;
ctx.lineCap = 'round';

// Обработчики событий
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function clearCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
```

2. **Загрузка примеров**:

```javascript
async function loadExample() {
    try {
        // Получаем список доступных примеров
        const response = await fetch('/examples/');
        const data = await response.json();
        
        // Выбираем случайный пример
        const randomExample = data.examples[Math.floor(Math.random() * data.examples.length)];
        const exampleUrl = `/static/test_digits/${randomExample}.jpg`;
        
        // Загружаем изображение
        const imgResponse = await fetch(exampleUrl);
        const blob = await imgResponse.blob();
        
        // Отображаем в текущем режиме
        if (drawSection.style.display !== 'none') {
            // В режиме рисования
            const img = await createImageBitmap(blob);
            clearCanvas();
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        } else {
            // В режиме загрузки
            preview.src = URL.createObjectURL(blob);
            preview.style.display = 'block';
            fileInput.files = [new File([blob], `example_${randomExample}.jpg`)];
        }
    } catch (error) {
        showResult(`Ошибка загрузки примера: ${error.message}`, false);
    }
}
```

3. **Отправка данных на сервер**:

```javascript
async function sendToServer(imageBlob) {
    const formData = new FormData();
    formData.append('file', imageBlob, 'digit.png');
    
    try {
        const response = await fetch('/predict/', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error(await response.text());
        
        const data = await response.json();
        showResult(`Цифра: <strong>${data.digit}</strong><br>
                  Уверенность: ${(data.confidence * 100).toFixed(1)}%<br>
                  Время: ${data.processing_time.toFixed(3)} сек`, true);
    } catch (error) {
        showResult("Ошибка распознавания", false);
    }
}
```

## Взаимодействие компонентов

1. **Пользовательский сценарий**:
   - Пользователь рисует цифру или загружает изображение
   - Клиент отправляет изображение на `/predict/`
   - Сервер обрабатывает запрос и возвращает результат
   - Клиент отображает результат

2. **Формат запроса/ответа**:
   - Запрос: `POST /predict/` с изображением в форме
   - Ответ: JSON с полями:
     ```json
     {
         "digit": 5,
         "confidence": 0.98,
         "processing_time": 0.12
     }
     ```

3. **Обработка ошибок**:
   - Неверный тип файла (400)
   - Ошибка обработки изображения (500)
   - Проблемы с моделью (500)

## Развертывание

### Локальный запуск

```bash
uvicorn main:app --reload
```

Приложение будет доступно по адресу: `http://localhost:8000`

### Docker-развертывание

1. Создайте `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Сборка и запуск:

```bash
docker build -t mnist-api .
docker run -d -p 8000:8000 mnist-api
```

### Развертывание на сервере

1. Установите nginx:
```bash
sudo apt install nginx
```

2. Настройте проксирование в nginx:
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. Запустите приложение с помощью systemd:
```bash
sudo systemctl start mnist-api
```

## Тестирование

### 1. Тестирование API

Через Swagger UI (`/docs`):
- Откройте `http://localhost:8000/docs`
- Протестируйте endpoint `/predict/`

Через cURL:
```bash
curl -X POST -F "file=@test.png" http://localhost:8000/predict/
```

### 2. Тестирование интерфейса

1. Проверьте рисование на canvas
2. Протестируйте загрузку изображений
3. Убедитесь, что кнопка "Пример" работает
4. Проверьте отображение результатов

### 3. Нагрузочное тестирование

С помощью `locust`:
```python
from locust import HttpUser, task, between

class MNISTUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict(self):
        with open("test.png", "rb") as f:
            self.client.post("/predict/", files={"file": f})
```

Запуск:
```bash
locust -f locustfile.py
```

## Возможные проблемы и решения

1. **Ошибка загрузки модели**:
   - Проверьте путь к файлу модели
   - Убедитесь, что версии TensorFlow совместимы
   - Решение: пересохраните модель с помощью `model.save()`

2. **Проблемы с CORS**:
   - Проверьте настройки CORS в коде
   - Убедитесь, что клиент обращается к правильному URL
   - Решение: явно указать разрешенные домены

3. **Неправильное распознавание**:
   - Проверьте предобработку изображения
   - Убедитесь, что изображение соответствует формату MNIST
   - Решение: добавить логирование промежуточных результатов

4. **Медленная работа**:
   - Проверьте загрузку CPU/GPU
   - Убедитесь, что используется TensorFlow с GPU поддержкой
   - Решение: оптимизировать размер изображения перед отправкой

## Дальнейшее развитие

1. **Улучшение модели**:
   - Добавление более сложной архитектуры CNN
   - Дообучение на дополнительных данных
   - Реализация ensemble моделей

2. **Расширение API**:
   - Добавление аутентификации
   - Реализация batch-обработки
   - Добавление rate limiting

3. **Улучшение интерфейса**:
   - Адаптивный дизайн
   - История запросов
   - Визуализация уверенности модели

4. **Мониторинг**:
   - Логирование запросов
   - Метрики производительности
   - Система оповещений

## Заключение

В этом руководстве мы рассмотрели полный цикл разработки системы распознавания рукописных цифр - от настройки окружения до развертывания готового решения. Проект демонстрирует современные подходы к созданию ML-приложений с веб-интерфейсом.

Ключевые преимущества реализации:
- Чистая архитектура с разделением ответственности
- Современные технологии (FastAPI, TensorFlow 2.x)
- Интерактивный интерфейс с поддержкой touch-устройств
- Гибкость для дальнейшего развития

Данное решение может служить основой для более сложных систем компьютерного зрения и обработки изображений.