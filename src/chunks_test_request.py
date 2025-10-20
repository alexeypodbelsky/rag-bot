from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Загрузка FAISS индекса
index_path = "faiss_index"  # Путь к сохраненному индексу
model_name = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=model_name)
db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# 2. Поиск по запросу
query = "Что известно о школе?"
results = db.similarity_search_with_score(
    query, k=30
)  # Или db.similarity_search_with_score(query)

# 3. Вывод результатов
print(f"Результаты поиска по запросу: '{query}'")
for i, (doc, score) in enumerate(results):
    print(f"Результат {i+1}:")
    print(f"  Текст: {doc.page_content}")
    print(f"  Метаданные: {doc.metadata}")
    print(f"  Score: {score}")  # Вывод score
    print("-" * 20)
