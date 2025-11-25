# Используется модель GPT-4.1 (Cascade)
# Температура генерации: 0.1
# Системный промпт для анализа рукописного текста
from dotenv import load_dotenv
load_dotenv()
import os
import mimetypes
import base64
from openai import OpenAI

MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.1

client = OpenAI()

HANDWRITER_SYSTEM_PROMPT = """Ты ассистент по распознаванию и анализу рукописного текста.
Твои задачи:
1) Аккуратно считывать текст с рукописных документов.
2) Пытаться понять структуру документа и извлечь из него основные данные.
3) Возвращать результат строго в формате JSON.

Под "основными данными" подразумеваются:
- реквизиты (имена, фамилии, номера телефонов, email, адреса);
- даты, суммы, валюты;
- заголовок документа, номер документа;
- табличные данные (списки позиций, строк таблицы и т.п.);
- любые явно важные значения, встречающиеся в документе.

Структура итогового JSON (примерная, можно не использовать все поля, если данных нет):
{
  "document_type": "...",        // тип документа, если удаётся понять (например: "заявление", "квитанция", "анкета", "заметка")
  "title": "...",                // заголовок, если есть
  "people": [
    {
      "full_name": "...",
      "role": "...",             // например: "отправитель", "получатель", "подписант"
      "contacts": {
        "phone": "...",
        "email": "...",
        "address": "..."
      }
    }
  ],
  "dates": [
    {
      "label": "...",            // например: "дата составления", "дата рождения"
      "value": "YYYY-MM-DD"      // по возможности нормализуй; если не получается, оставь исходный формат
    }
  ],
  "amounts": [
    {
      "label": "...",
      "value": number,
      "currency": "..."
    }
  ],
  "fields": {                    // произвольные ключ-значение пары по документу
    "ключ": "значение"
  },
  "tables": [
    {
      "name": "...",             // название таблицы, если есть
      "columns": ["...", "..."],
      "rows": [
        ["значение1", "значение2"],
        ["...", "..."]
      ]
    }
  ],
  "raw_text": "полный распознанный текст документа одной строкой"
}

В ответе НЕЛЬЗЯ добавлять никакие пояснения вне JSON.
НЕ пиши текст до или после JSON.
"""

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

def analyze_handwriter_image(image_path: str) -> str:
    """Анализирует изображение с рукописным текстом с помощью OpenAI API"""
    # Кодируем изображение
    image_data_url = encode_image_to_data_url(image_path)
    
    # Отправляем запрос к OpenAI API
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": HANDWRITER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Проанализируй изображение с рукописным текстом и верни информацию в формате JSON согласно инструкциям."},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ],
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Пример использования
    image_path = input("Введите путь к изображению с рукописным текстом: ").strip()
    if image_path:
        try:
            print("Анализ рукописного текста...")
            result = analyze_handwriter_image(image_path)
            print(result)
        except Exception as e:
            print(f"Ошибка при анализе изображения: {e}")
    else:
        print("Путь к изображению не указан.")