import json
import os
import re


def replace_text_in_files(terms_map_path, original_dir, processed_dir):
    """
    Заменяет текст в файлах из original_dir согласно словарю из terms_map_path,
    игнорируя регистр при сравнении. Сохраняет измененные файлы в processed_dir.

    Args:
        terms_map_path (str): Путь к файлу JSON со словарем замен.
        original_dir (str): Путь к директории с оригинальными файлами.
        processed_dir (str): Путь к директории для сохранения обработанных файлов.
    """

    # Загрузка словаря замен из JSON-файла
    with open(terms_map_path, "r", encoding="utf-8") as f:
        terms_map = json.load(f)

    # Создание директории для обработанных файлов, если она не существует
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Перебор файлов в директории с оригинальными файлами
    for filename in os.listdir(original_dir):
        if os.path.isfile(os.path.join(original_dir, filename)):
            # Полный путь к оригинальному файлу
            original_filepath = os.path.join(original_dir, filename)
            # Полный путь к файлу, в который будет сохранен обработанный файл
            processed_filepath = os.path.join(processed_dir, filename)

            try:
                # Чтение содержимого оригинального файла
                with open(original_filepath, "r", encoding="utf-8") as f:
                    file_content = f.read()

                # Замена текста в содержимом файла согласно словарю
                for key, value in terms_map.items():
                    # Используем re.sub с флагом re.IGNORECASE для игнорирования регистра
                    file_content = re.sub(
                        re.escape(key), value, file_content, flags=re.IGNORECASE
                    )

                # Сохранение измененного содержимого в новый файл
                with open(processed_filepath, "w", encoding="utf-8") as f:
                    f.write(file_content)

                print(
                    f"Файл '{filename}' успешно обработан и сохранен в '{processed_dir}'"
                )

            except Exception as e:
                print(f"Ошибка при обработке файла '{filename}': {e}")


if __name__ == "__main__":
    # Задание путей
    terms_map_path = "terms_map.json"  # Путь к файлу со словарем
    original_dir = "original"  # Папка с оригинальными файлами
    processed_dir = "processed"  # Папка для обработанных файлов

    # Вызов функции для замены текста в файлах
    replace_text_in_files(terms_map_path, original_dir, processed_dir)
