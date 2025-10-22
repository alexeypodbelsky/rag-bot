import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline  #  Изменено
import torch


# 1. Загрузка FAISS индекса
def load_faiss_index(index_path, embeddings_model_name):
    """Загружает FAISS индекс из указанного пути."""
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return db


# 3. Инициализация Flan-T5 модели из локальной папки
def initialize_flan_t5_local(model_path):
    """Инициализирует модель Flan-T5 из локальной папки."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=550,
    )

    return pipe


def initialize_flan_t5_huggingface():
    model_name = "google/flan-t5-base"
    print(f"Загрузка модели из Hugging Face: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=550,
    )

    return pipe


# 4. Формирование промпта
def create_prompt(
    user_query, search_results, few_shot_examples="", chain_of_thought_prefix=""
):
    """Формирует промпт для LLM, объединяя запрос пользователя и результаты поиска."""
    prompt = f"""{chain_of_thought_prefix}
{few_shot_examples}

[User]: {user_query}

{search_results}

"""
    return prompt


# 5. Генерация ответа с помощью LLM
def generate_answer(prompt, flan_t5_pipeline):
    """Генерирует ответ с использованием LLM."""
    result = flan_t5_pipeline(prompt)
    return result[0]["generated_text"]


# Основная функция
def main():
    # Настройки
    index_path = "faiss_index"  # Путь к индексу FAISS
    embeddings_model_name = (
        "sentence-transformers/all-MiniLM-L6-v2"  # Модель эмбеддингов
    )

    # flan_t5_model_path = "flan-t5-base"  #  Путь к локальной папке с Flan-T5
    flan_t5_model_path = ""  #  Путь к локальной папке с Flan-T5

    num_search_results = 3  # кол-во чанков, которые достаем из векторной базы

    # 1. Загружаем индекс
    db = load_faiss_index(index_path, embeddings_model_name)

    # 2. Загружаем  модель Flan-T5
    if len(flan_t5_model_path) != 0:
        flan_t5_pipeline = initialize_flan_t5_local(flan_t5_model_path)
    else:
        flan_t5_pipeline = initialize_flan_t5_huggingface()

    #  Примеры Few-shot
    few_shot_examples = """[Example]Q: how I can walk the Archive?
A: is on the left in The Archive, or on the right.
"""

    # Chain-of-Thought
    chain_of_thought_prefix = "[System] You are an assistant that thinks first, then answers. Always write your steps. If you don't know the answer, just say: 'I don't know'"

    # Бесконечный цикл обработки запросов
    while True:
        user_query = input("Введите запрос (или 'exit' для выхода): ")
        if user_query.lower() == "exit":
            break

        # 3. Ищем похожие чанки в векторной базе
        results = db.similarity_search(user_query, k=num_search_results)
        search_results_text = "\n".join([doc.page_content for doc in results])

        # 5. Формируем промпт
        prompt = create_prompt(
            user_query, search_results_text, few_shot_examples, chain_of_thought_prefix
        )

        print(prompt)

        # 6. Генерируем ответ
        answer = generate_answer(prompt, flan_t5_pipeline)

        # 7. Выводим ответ
        print("Ответ LLM:")
        print(answer)
        print("-" * 40)


if __name__ == "__main__":
    main()
