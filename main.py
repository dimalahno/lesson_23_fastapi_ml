from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import requests
from io import BytesIO
from PIL import Image
import numpy as np

# Импортируем нужные части TensorFlow/Keras для VGG19
from tensorflow.keras.applications.vgg19 import (
    VGG19, 
    decode_predictions, 
    preprocess_input
)
from tensorflow.keras.preprocessing.image import img_to_array

# Создаём приложение
app = FastAPI()

# Загружаем предобученную модель VGG19 с весами ImageNet
model = VGG19(weights="imagenet")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """
    Отдаём пользователю содержимое файла index.html (форма для ввода ссылки).
    """
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/predict", response_class=HTMLResponse)
async def predict_image(request: Request, image_url: str = Form(...)):
    """
    Принимаем POST-запрос со ссылкой на изображение, загружаем его,
    пропускаем через модель VGG19 и возвращаем результат классификации.
    """
    # 1. Пытаемся скачать изображение по URL
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Генерирует ошибку, если статус не 200
    except Exception as e:
        return f"""
        <html>
            <body>
                <h2>Ошибка при загрузке изображения: {e}</h2>
                <a href="/">Вернуться назад</a>
            </body>
        </html>
        """

    # 2. Пытаемся открыть изображение с помощью Pillow
    try:
        img = Image.open(BytesIO(response.content))
    except Exception as e:
        return f"""
        <html>
            <body>
                <h2>Не удалось открыть изображение: {e}</h2>
                <a href="/">Вернуться назад</a>
            </body>
        </html>
        """

    # 3. Подготовка изображения для VGG19: 
    #    меняем размер на 224x224, превращаем в массив и нормализуем
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # приведение к формату, которого ожидает VGG19

    # 4. Запрашиваем у модели предсказание
    preds = model.predict(x)
    # decode_predictions вернёт список списков; возьмём top=3
    top_preds = decode_predictions(preds, top=3)[0]
    # Например, берем самое вероятное предсказание
    best_label = top_preds[0][1]           # Имя класса
    best_confidence = top_preds[0][2]      # Вероятность
    # Для наглядности соберём все три результата
    all_preds = "<br/>".join([
        f"{p[1]} (вероятность: {p[2]:.2f})" for p in top_preds
    ])

    # 5. Формируем HTML-ответ
    # Отобразим исходный URL, само изображение и результаты классификации
    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Результат классификации</title>
    </head>
    <body>
        <h2>Классификация изображения на VGG19</h2>
        <form action="/predict" method="post">
            <label for="image_url">Ссылка на изображение:</label><br/>
            <input type="text" id="image_url" name="image_url" size="50" placeholder="Введите URL"/>
            <button type="submit">Отправить</button>
        </form>

        <hr/>

        <h3>Вы ввели ссылку:</h3>
        <p><a href="{image_url}" target="_blank">{image_url}</a></p>

        <img src="{image_url}" alt="Изображение не найдено" style="max-width:400px;"/>

        <h3>Результаты распознавания (Top-1):</h3>
        <p><b>{best_label}</b> (вероятность: {best_confidence:.2f})</p>

        <h3>Топ-3 предсказания:</h3>
        <p>{all_preds}</p>
    </body>
    </html>
    """

