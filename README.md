# Nori — 2D-аватар с текстовым управлением

Аватар в pygame, который реагирует на русские реплики мимикой и короткими
анимациями. Решение о реакции принимает LLM в фоновом потоке — интерфейс
не подвисает.

Провайдеры (по порядку):

1. **Groq** (облако, бесплатный тариф) — если в `.env` есть `GROQ_API_KEY`.
   По умолчанию `qwen/qwen3.8-27b`, ответ за 0.3–1 с.
2. **Ollama** (локальная Qwen 2.5) — если ключа нет или Groq недоступен.
3. Локальный семантический разбор — если недоступно всё остальное.

## Запуск

```bash
cd ~/Проекты/Nori
.venv-avatar/bin/python main.py
```

Ключ Groq кладётся в `.env` рядом с `main.py` (файл в `.gitignore`):

```
GROQ_API_KEY=gsk_...
NORI_LLM_PROVIDER=auto      # auto | groq | ollama
NORI_GROQ_MODEL=qwen/qwen3.8-27b
```

Модель Groq можно посмотреть так:

```bash
curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
```

Для локального варианта нужен запущенный Ollama:

```bash
ollama serve && ollama pull qwen2.5:3b
```

Переопределение: `NORI_QWEN_MODEL=qwen2.5:1.5b`,
`NORI_OLLAMA=http://127.0.0.1:11434`.

## Управление

- `Tab` / `1` `2` `3` — переключение аватаров
- `F2` — нейтральная поза
- `H` — режим чёлки (полная / правая прядь / без чёлки)
- `Esc` — выход

## Проверка

```bash
.venv-avatar/bin/python selftest.py
```

Скрипт проверяет роутер, таймлайны и рендерит кадры в `_preview/`.

## Структура

- `main.py` — игровой цикл, аватары Nori / Classic / Smiley
- `qwen_client.py` — асинхронный клиент Ollama + валидация ответов
- `semantic_router.py` — локальный резервный разбор и кэш
- `expressions.py` — пресеты эмоций
- `timeline.py` — keyframe-анимации
- `export_psd.py` — экспорт слоёв из `py.psd`
