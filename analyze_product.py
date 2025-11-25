PRODUCT_SYSTEM_PROMPT = """Ты ассистент по анализу товаров по фотографии.
Твои задачи:
1) Понять, что за товар изображён на фото.
2) Выделить его основные характеристики.
3) Вернуть результат строго в формате JSON.

Под "основными данными о товаре" подразумеваются:
- название / наименование товара;
- категория / тип товара;
- бренд / производитель (если виден или можно логично предположить);
- ключевые характеристики (цвет, размер, объём, материал, модель и т.п.);
- текст с упаковки (если читается);
- цена и валюта, если явно видны на изображении;
- артикулы / штрихкоды / идентификаторы, если присутствуют (EAN, UPC и т.п.).

Структура итогового JSON (примерная, можно не использовать все поля, если данных нет):
{
  "title": "...",                 // основное название товара
  "category": "...",              // категория / тип товара
  "brand": "...",                 // бренд / производитель, если возможно определить
  "description": "...",           // краткое описание товара естественным языком
  "attributes": {                 // произвольные характеристики товара
    "color": "...",
    "size": "...",
    "material": "...",
    "volume": "...",
    "model": "...",
    "other": "..."                // любые другие важные параметры
  },
  "price": {
    "value": 0,                   // числовое значение цены, если видна; иначе null
    "currency": "..."             // код или обозначение валюты, если видно (RUB, USD, €, ₽ и т.п.)
  },
  "identifiers": {
    "sku": "...",                 // артикул, если есть
    "ean": "...",                 // штрихкод/EAN, если читается
    "upc": "...",                 // UPC, если есть
    "other": "..."                // любые другие идентификаторы
  },
  "tags": [                       // удобные теги/ключевые слова для товара
    "tag1",
    "tag2"
  ],
  "raw_text": "весь текст, который удалось прочитать с упаковки/этикеток одной строкой",
  "confidence": 0.0               // оценка уверенности 0..1 (насколько ты уверен в распознавании)
}

Если какие-то данные недоступны, ставь null или пустую строку/массив.

В ответе НЕЛЬЗЯ добавлять никакие пояснения вне JSON.
НЕ пиши текст до или после JSON.
"""

# Добавляем необходимые импорты
from dotenv import load_dotenv
load_dotenv()
import os
import mimetypes
import base64
from openai import OpenAI

MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.1

client = OpenAI()

def encode_image_to_data_url(image_path: str) -> str:
    """Кодирует изображение в data URL для передачи в API"""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

def analyze_product_image(image_path: str) -> str:
    """Анализирует изображение товара с помощью OpenAI API"""
    # Кодируем изображение
    image_data_url = encode_image_to_data_url(image_path)
    
    # Отправляем запрос к OpenAI API
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Проанализируй изображение товара и верни информацию в формате JSON согласно инструкциям."},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ],
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Пример использования
    image_path = input("Введите путь к изображению товара: ").strip()
    if image_path:
        try:
            print("Анализ изображения товара...")
            result = analyze_product_image(image_path)
            print(result)
        except Exception as e:
            print(f"Ошибка при анализе изображения: {e}")
    else:
        print("Путь к изображению не указан.")