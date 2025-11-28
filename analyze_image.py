# Системный промпт берётся из файла system_prompt.py
from dotenv import load_dotenv
load_dotenv()
import os
import mimetypes
import base64
from typing import Optional
from docx import Document
from openai import OpenAI
from system_prompt import SYSTEM_PROMPT

MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.1

client = OpenAI()

# Пример использования OpenAI API с явным указанием модели, температуры и системного промпта
def ask_gpt(user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content

def encode_image_to_data_url(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

def extract_text_from_docx(docx_path: str) -> str:
    if not os.path.isfile(docx_path):
        raise FileNotFoundError(f"Файл не найден: {docx_path}")
    doc = Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

if __name__ == "__main__":
    docx_path = r"C:\projects\PythonProject\Резюме (Python- разработчик) 3.DOCX"
    print("Анализ документа...")
    text = extract_text_from_docx(docx_path)
    result = ask_gpt(text)
    print(result)
