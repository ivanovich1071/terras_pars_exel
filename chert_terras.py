import os
import cv2
import pytesseract
import easyocr
import pandas as pd
from datetime import datetime

# Укажите путь к установленному Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image):
    """
    Улучшенная предобработка изображения для OCR.
    """
    # Увеличение контраста
    alpha = 2.0  # Контраст
    beta = -30  # Яркость
    contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Фильтрация шума
    blurred = cv2.GaussianBlur(contrast, (5, 5), 0)

    # Бинаризация
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def extract_text_with_tesseract(image_path):
    """
    Извлечение текста с помощью Tesseract OCR.
    """
    # Чтение изображения
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return ""

    # Предобработка изображения
    preprocessed_image = preprocess_image(image)

    # Настройка параметров Tesseract
    custom_config = r'--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ°±´`"\'.,-_/(){}[]|'

    # Распознавание текста
    text = pytesseract.image_to_string(preprocessed_image, lang='eng+rus', config=custom_config)
    return text


def extract_text_with_easyocr(image_path):
    """
    Извлечение текста с помощью EasyOCR.
    """
    reader = easyocr.Reader(['ru', 'en'])
    results = reader.readtext(image_path)
    text = "\n".join([res[1] for res in results])
    return text


def segment_image(image):
    """
    Разделение изображения на строки или блоки текста.
    """
    # Инверсия изображения
    binary = cv2.bitwise_not(image)

    # Морфологическая обработка для выделения строк
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    line_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10:  # Исключаем слишком мелкие строки
            segments.append(image[y:y + h, x:x + w])
    return segments


def parse_text(text):
    """
    Разбор текста в структурированный формат.
    """
    data = [{'Text': line.strip()} for line in text.split('\n') if line.strip()]
    return data


def save_to_excel(parsed_data, output_path):
    """
    Сохранение структурированных данных в Excel.
    """
    if not parsed_data:
        print("Нет данных для сохранения.")
        return

    # Удаление существующего файла, если он есть
    if os.path.exists(output_path):
        os.remove(output_path)

    # Создание нового файла
    df = pd.DataFrame(parsed_data)
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Данные успешно сохранены в файл: {output_path}")


def main():
    input_folder = "chert"  # Папка с изображениями
    output_folder = "output"  # Папка для сохранения результатов
    os.makedirs(output_folder, exist_ok=True)

    # Список изображений
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Нет файлов изображений в папке {input_folder}.")
        return

    for image_file in image_files:
        image_path = os.path.abspath(os.path.join(input_folder, image_file))
        print(f"Обрабатывается файл: {image_file}")

        # Извлечение текста с Tesseract
        tesseract_text = extract_text_with_tesseract(image_path)

        # Извлечение текста с EasyOCR
        easyocr_text = extract_text_with_easyocr(image_path)

        # Объединение результатов OCR
        combined_text = f"Tesseract OCR:\n{tesseract_text}\n\nEasyOCR:\n{easyocr_text}"

        # Разбор текста
        parsed_data = parse_text(combined_text)

        # Сохранение в Excel
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.xlsx")
        save_to_excel(parsed_data, output_path)


if __name__ == "__main__":
    main()
