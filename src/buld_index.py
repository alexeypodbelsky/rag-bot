import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader

# 1. Загрузка документов
data_folder = "../knowledge_base/processed"  # Папка с текстовыми файлами
documents = []
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_folder, filename)
        loader = TextLoader(filepath)
        documents.extend(loader.load())


# 2. Разбиение на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=40
)  # Настройка размера чанков и перекрытия
chunks = text_splitter.split_documents(documents)

time_start_s = time.time()  # Начальный момент времени генерации

# 3. Выбор и инициализация модели Sentence Transformers
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Модель из Hugging Face
embeddings = HuggingFaceEmbeddings(model_name=model_name)

embedding_size = embeddings.client.get_sentence_embedding_dimension()
print(f"Размерность эмбеддинга: {embedding_size}")

# 4. Создание векторной базы данных FAISS и добавление эмбеддингов
db = FAISS.from_documents(chunks, embeddings)

time_end_s = time.time()  # Конечный момент времени генерации
elapsed_time = time_end_s - time_start_s

# 5. Сохранение FAISS индекса (чтобы потом можно было загрузить)
db.save_local("faiss_index")

print(f"Создано {len(chunks)} чанков.")
print("Индекс FAISS создан и сохранен в 'faiss_index'")
print(f"Время выполнения генерации: {elapsed_time} секунд")
