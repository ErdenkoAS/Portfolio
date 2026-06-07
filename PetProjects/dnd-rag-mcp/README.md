# Домашнее задание: RAG + MCP — База знаний D&D

**Курс:** Промышленная разработка ИИ-агентов  
**Тема:** RAG и протоколы взаимодействия ИИ-агентов

## Что реализовано

Полный пайплайн RAG + MCP на основе Книги игрока D&D 5e (PHB, русский перевод):

```
PDF (331 стр.) → чанки + метаданные → Qdrant (3182 вектора)
                                              ↓
                                    mcp-server-qdrant (http)
                                              ↓
                                    Агент LangChain + GigaChat
                                              ↓
                                    top-k результатов с метаданными
```

## Структура проекта

```
hw_4/
├── docs/
│   └── phb_ru.pdf              # Книга игрока D&D (русский перевод)
├── qdrant_storage/             # Локальная векторная БД
├── 01_ingest.ipynb             # Загрузка PDF, чанкинг, индексация
├── 02_search_demo.ipynb        # Поиск через MCP-инструмент
├── 03_agent_demo.ipynb         # Агент LangChain + GigaChat с MCP
├── 04_eval.ipynb               # Оценка качества поиска (15 запросов)
├── eval_queries.csv            # Результаты оценки
├── mcp_config.json             # Конфигурация MCP-сервера
├── requirements.txt            # Зависимости Python
├── .env.example                # Шаблон переменных окружения
└── .env                        # Ваши ключи (не коммитить!)
```

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
pip install mcp-server-qdrant
```

### 2. Настройка переменных окружения

```bash
cp .env.example .env
```

Заполните `.env`:

```
GIGACHAT_CREDENTIALS=<ваш_Base64_client_id:client_secret>
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_MODEL=GigaChat-2-Max
QDRANT_PATH=./qdrant_storage
QDRANT_COLLECTION=dnd_phb
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### 3. Подготовка корпуса

Положите PDF в папку `docs/`:

```
docs/phb_ru.pdf
```

### 4. Запуск индексации

Откройте и выполните все ячейки в `01_ingest.ipynb`. Ноутбук выполнит:
- Чтение PDF (316 страниц с текстом)
- Разбивку на 3182 чанка по 500 символов
- Сохранение метаданных: `chunk_id`, `document_id`, `source`, `page`
- Загрузку в Qdrant (коллекция `dnd_phb` через GigaChat Embeddings)
- Загрузку в Qdrant (коллекция `dnd_phb_mcp` через fastembed для MCP-сервера)

### 5. Запуск MCP-сервера

В отдельном терминале:

```bash
QDRANT_LOCAL_PATH="<абсолютный_путь>/hw_4/qdrant_storage" \
COLLECTION_NAME="dnd_phb_mcp" \
EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
mcp-server-qdrant --transport streamable-http
```

Сервер запустится на `http://127.0.0.1:8000/mcp`.

### 6. Демонстрационный поиск через MCP

Выполните `02_search_demo.ipynb`.

### 7. Запуск агента

Выполните `03_agent_demo.ipynb`.

### 8. Оценка качества поиска

Выполните `04_eval.ipynb` — прогоняет 15 запросов и сохраняет результаты в `eval_queries.csv`.

## MCP-сервер

| Параметр | Значение |
|---|---|
| Сервер | mcp-server-qdrant |
| Транспорт | streamable-http |
| URL | http://127.0.0.1:8000/mcp |
| Инструмент поиска | qdrant-find |
| Коллекция | dnd_phb_mcp |
| Модель эмбеддингов | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |

## Корпус документов

| Параметр | Значение |
|---|---|
| Источник | Книга игрока D&D 5e (PHB, русский перевод) |
| Страниц | 331 (316 с текстом) |
| Чанков | 3182 |
| Размер чанка | 500 символов |
| Перекрытие | 50 символов |

Метаданные каждого чанка: `document_id`, `chunk_id`, `source`, `page`.

## Результаты оценки

15 запросов разных типов (см. `eval_queries.csv`):

| Оценка | Количество |
|---|---|
| хорошо | 5 |
| частично | 4 |
| плохо | 4 |
| ожидаемо (вне корпуса) | 2 |

## Стек

- **LLM:** GigaChat-2-Max
- **Эмбеддинги (RAG):** GigaChat Embeddings (1024)
- **Эмбеддинги (MCP):** fastembed paraphrase-multilingual-mpnet-base-v2 (768)
- **Векторная БД:** Qdrant (локально на диске)
- **MCP-сервер:** mcp-server-qdrant
- **Агент:** LangChain `create_agent` + `langchain-mcp-adapters`