FROM python:3.13

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем необходимые файлы и папки
COPY faiss_index ./faiss_index
COPY rag_bot_flan_t5_base.py .

# Делаем скрипт исполняемым
RUN chmod +x rag_bot_flan_t5_base.py

# Определяем команду для запуска скрипта
CMD ["python", "rag_bot_flan_t5_base.py"]

