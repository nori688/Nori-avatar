"""Гибридный семантический роутер (Semantic Vector Matcher + Negation/Compound Analyzer + LLM Client).

Обеспечивает мгновенный (<1 мс) семантический поиск по расширенной базе русскоязычных фраз,
разбирает отрицания («не злись»), составные команды («улыбнись и посмотри влево»)
и поддерживает динамический кэш с возможностью подключения локальной LLM.
"""

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import expressions

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_FILE = PROJECT_DIR / "models" / "semantic_cache.json"

# База образцов фраз, привязанных к пресетам и их интенсивности
CORPUS: List[Tuple[str, List[Tuple[str, float]]]] = [
    # Радость / Улыбка
    ("улыбнись", [("smile", 0.85)]),
    ("улыбайся", [("smile", 0.80)]),
    ("сделай улыбку", [("smile", 0.90)]),
    ("покажи радость", [("smile", 0.80)]),
    ("я очень рад", [("smile", 0.75)]),
    ("мне весело", [("smile", 0.80)]),
    ("отличная новость", [("smile", 0.70)]),
    ("рад тебя видеть", [("smile", 0.85)]),
    ("ура", [("joy", 0.90)]),
    ("кайф", [("joy", 0.85)]),
    ("супер", [("smile", 0.75)]),
    ("красота", [("smile", 0.70)]),
    ("ты прелесть", [("smile", 0.85)]),
    ("замечательно", [("smile", 0.75)]),
    ("я счастлив", [("joy", 0.90)]),

    # Смех / Восторг
    ("рассмейся", [("laugh", 0.95)]),
    ("посмейся", [("laugh", 0.85)]),
    ("хаха", [("laugh", 0.90)]),
    ("ахаха", [("laugh", 0.95)]),
    ("это очень смешно", [("laugh", 0.85)]),
    ("ржу", [("laugh", 0.95)]),
    ("ору", [("laugh", 1.00)]),
    ("угар", [("laugh", 0.90)]),
    ("восторг", [("joy", 0.95)]),
    ("я в восторге", [("joy", 0.95)]),

    # Грусть / Печаль
    ("грустно", [("sad", 0.85)]),
    ("мне грустно", [("sad", 0.90)]),
    ("покажи печаль", [("sad", 0.85)]),
    ("я расстроен", [("sad", 0.80)]),
    ("мне плохо", [("sad", 0.85)]),
    ("хочется плакать", [("sad", 0.95)]),
    ("обидно", [("sad", 0.75)]),
    ("тоска", [("sad", 0.85)]),
    ("печальное лицо", [("sad", 0.80)]),
    ("настроение на нуле", [("sad", 0.80)]),
    ("все плохо", [("sad", 0.90)]),
    ("сочувствую", [("sad", 0.65)]),

    # Злость / Гнев
    ("злись", [("angry", 0.90)]),
    ("злиться", [("angry", 0.90)]),
    ("не злись", [("neutral", 1.00), ("smile", 0.30)]),
    ("хватит злиться", [("neutral", 1.00), ("smile", 0.30)]),
    ("не сердись", [("neutral", 1.00), ("smile", 0.30)]),
    ("перестань дуться", [("neutral", 1.00), ("smile", 0.35)]),
    ("я злюсь", [("angry", 0.90)]),
    ("покажи злость", [("angry", 0.85)]),
    ("разозлись", [("angry", 0.95)]),
    ("нахмурься", [("angry", 0.75)]),
    ("хмуриться", [("angry", 0.75)]),
    ("грозный взгляд", [("angry", 0.85)]),
    ("меня это бесит", [("angry", 0.95)]),
    ("достали", [("angry", 0.85)]),
    ("я в ярости", [("angry", 1.00)]),
    ("сердитое лицо", [("angry", 0.80)]),
    ("бесишь", [("angry", 0.90)]),
    ("отвали", [("angry", 0.70), ("look_right", 0.40)]),

    # Удивление / Шок
    ("ого", [("surprise", 0.80)]),
    ("ничего себе", [("surprise", 0.90)]),
    ("вот это да", [("surprise", 0.85)]),
    ("удивись", [("surprise", 0.85)]),
    ("я удивлен", [("surprise", 0.80)]),
    ("глаза на лоб", [("shock", 0.95)]),
    ("шок", [("shock", 0.95)]),
    ("я в шоке", [("shock", 0.95)]),
    ("не может быть", [("surprise", 0.90)]),
    ("ты серьезно", [("skeptical", 0.85)]),
    ("обалдеть", [("surprise", 0.90)]),
    ("офигеть", [("shock", 0.90)]),

    # Подмигивание
    ("подмигни", [("wink_left", 0.90)]),
    ("подмигни левым глазом", [("wink_left", 0.95)]),
    ("подмигни правым глазом", [("wink_right", 0.95)]),
    ("сделай подмигивание", [("wink_left", 0.90)]),
    ("моргни одним глазом", [("wink_left", 0.85)]),

    # Сарказм / Ухмылка / Скептицизм
    ("ухмыльнись", [("smug", 0.90)]),
    ("хитрое лицо", [("smug", 0.85)]),
    ("я все знаю", [("smug", 0.80)]),
    ("коварная улыбка", [("smug", 0.90)]),
    ("сомневаюсь", [("skeptical", 0.85)]),
    ("да ладно", [("skeptical", 0.80)]),
    ("не верю", [("skeptical", 0.85)]),
    ("подозрительно", [("skeptical", 0.85)]),

    # Скука / Усталость / Сон
    ("мне скучно", [("bored", 0.85)]),
    ("скукотища", [("bored", 0.90)]),
    ("я устал", [("sleepy", 0.80)]),
    ("спать хочу", [("sleepy", 0.90)]),
    ("зеваю", [("sleepy", 0.75), ("talk", 0.70)]),
    ("лениво", [("bored", 0.80)]),

    # Взгляд и повороты головы
    ("посмотри налево", [("look_left", 0.90)]),
    ("глянь влево", [("look_left", 0.85)]),
    ("глаза налево", [("look_left", 0.95)]),
    ("смотри влево", [("look_left", 0.85)]),
    ("поверни голову налево", [("look_left", 0.95)]),

    ("посмотри направо", [("look_right", 0.90)]),
    ("глянь вправо", [("look_right", 0.85)]),
    ("глаза направо", [("look_right", 0.95)]),
    ("смотри вправо", [("look_right", 0.85)]),
    ("поверни голову направо", [("look_right", 0.95)]),

    ("посмотри вверх", [("look_up", 0.90)]),
    ("глянь наверх", [("look_up", 0.85)]),
    ("взгляни на потолок", [("look_up", 0.95)]),
    ("глаза вверх", [("look_up", 0.95)]),

    ("посмотри вниз", [("look_down", 0.90)]),
    ("глянь вниз", [("look_down", 0.85)]),
    ("взгляни под ноги", [("look_down", 0.95)]),
    ("глаза вниз", [("look_down", 0.95)]),
    ("опусти взгляд", [("look_down", 0.85)]),

    # Моргание / Спокойствие
    ("моргни", [("blink", 1.00)]),
    ("закрой глаза", [("blink", 1.00)]),
    ("зажмурься", [("blink", 1.00)]),
    ("открой глаза", [("neutral", 1.00)]),

    # Речь
    ("скажи что-нибудь", [("talk", 0.80)]),
    ("открой рот", [("talk", 0.85)]),
    ("говори", [("talk", 0.80)]),

    # Нейтраль
    ("нейтральное лицо", [("neutral", 1.00)]),
    ("смотри прямо", [("neutral", 1.00)]),
    ("успокойся", [("neutral", 1.00)]),
    ("расслабься", [("neutral", 1.00)]),
    ("стой спокойно", [("neutral", 1.00)]),
    ("вернись в центр", [("neutral", 1.00)]),
]

# Правила для отрицаний: если есть «не» / «хватит» перед эмоцией
NEGATION_RULES = {
    "angry": [("neutral", 1.0), ("smile", 0.35)],  # не злись -> спокойно + легкая улыбка
    "sad": [("smile", 0.60)],                       # не грусти -> приободрись
    "smile": [("neutral", 1.0)],                    # не улыбайся -> нейтраль / серьёзное
    "joy": [("neutral", 1.0)],                      # хватит ржать -> нейтраль
    "laugh": [("neutral", 1.0)],
    "look_left": [("neutral", 1.0)],
    "look_right": [("neutral", 1.0)],
    "blink": [("neutral", 1.0)],
}


class SemanticRouter:
    """Семантический векторный маршрутизатор с разбором отрицаний и кэшированием."""

    def __init__(self):
        self.phrases = [p[0] for p in CORPUS]
        self.targets = [p[1] for p in CORPUS]

        # Символьные и пословные n-граммы (char_wb 2-4) дают устойчивость к окончаниям и опечаткам
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.phrases)

        # Загрузка постоянного семантического кэша
        self.cache: Dict[str, List[Tuple[str, float]]] = {}
        self._load_cache()

    def _load_cache(self):
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.cache = {k: [tuple(x) for x in v] for k, v in raw.items()}
            except Exception:
                self.cache = {}

    def _save_cache(self):
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _match_single_clause(self, clause: str) -> Tuple[List[Tuple[str, float]], float]:
        """Ищет совпадение для одного простого предложения/клаузы."""
        clause = clause.strip().lower()
        if not clause:
            return [("neutral", 1.0)], 1.0

        # 1. Проверка прямого кэша
        if clause in self.cache:
            return self.cache[clause], 1.0

        # 2. Проверка отрицания («не злись», «хватит грустить», «перестань улыбаться»)
        is_negated = bool(re.search(r"\b(не|хватит|перестань|прекрати)\b", clause))
        cleaned = re.sub(r"\b(не|хватит|перестань|прекрати|пожалуйста|давай|ты)\b", "", clause).strip()
        search_target = cleaned if (is_negated and cleaned) else clause

        # 3. Векторный поиск по косинусному сходству
        vec = self.vectorizer.transform([search_target])
        sims = cosine_similarity(vec, self.tfidf_matrix)[0]
        best_idx = int(np.argmax(sims))
        score = float(sims[best_idx])

        matched_presets = self.targets[best_idx]

        # Если было отрицание, преобразуем пресет согласно правилам анти-эмоций
        if is_negated:
            inverted = []
            for name, _ in matched_presets:
                if name in NEGATION_RULES:
                    inverted.extend(NEGATION_RULES[name])
                else:
                    inverted.append(("neutral", 1.0))
            matched_presets = inverted or [("neutral", 1.0)]

        return matched_presets, score

    def route(self, text: str) -> Tuple[Dict[str, float], str, float]:
        """Разбирает составной текст, находит пресеты и возвращает итоговые параметры."""
        norm_text = text.strip().lower()
        if not norm_text:
            return expressions.NEUTRAL.copy(), "Пустой ввод", 1.0

        # Проверка кэша для всей фразы целиком
        if norm_text in self.cache:
            blended = expressions.blend_presets(self.cache[norm_text])
            return blended, f"Кэш: {self.cache[norm_text]}", 1.0

        # Разбор составных фраз (разделители: «и», «а», «затем», «потом», запятая)
        parts = re.split(r"[,;]|\b(и|а|затем|потом|после чего)\b", norm_text)
        clauses = [p.strip() for p in parts if p and p.strip() not in ("и", "а", "затем", "потом", "после чего")]

        combined_presets: List[Tuple[str, float]] = []
        scores: List[float] = []

        for c in clauses:
            presets, score = self._match_single_clause(c)
            combined_presets.extend(presets)
            scores.append(score)

        avg_score = float(np.mean(scores)) if scores else 0.0

        # Если уверенность низкая, пробуем опросить локальную LLM (Ollama), если она запущена
        if avg_score < 0.45:
            llm_presets = self._try_llm_query(norm_text)
            if llm_presets:
                combined_presets = llm_presets
                avg_score = 0.99
                self.cache[norm_text] = combined_presets
                self._save_cache()
                blended_params = expressions.blend_presets(combined_presets)
                model_name = getattr(self, "_active_model", "Qwen")
                return blended_params, f"LLM ({model_name}): {combined_presets}", 0.99

        # Если уверенность достаточна, сохраняем в кэш
        if avg_score >= 0.40:
            self.cache[norm_text] = combined_presets
            self._save_cache()

        blended_params = expressions.blend_presets(combined_presets)
        description = f"Семантика ({avg_score:.2f}): " + ", ".join(f"{p}:{w:.2f}" for p, w in combined_presets)

        return blended_params, description, avg_score

    def _detect_ollama_model(self) -> str:
        """Определяет лучшую доступную модель Qwen в Ollama."""
        if hasattr(self, "_active_model") and self._active_model:
            return self._active_model

        try:
            import urllib.request
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=0.6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                for priority in ("qwen2.5:7b", "qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5:0.5b", "qwen2.5", "llama3"):
                    for m in models:
                        if priority in m:
                            self._active_model = m
                            return m
                if models:
                    self._active_model = models[0]
                    return models[0]
        except Exception:
            pass

        self._active_model = "qwen2.5:3b"
        return self._active_model

    def _try_llm_query(self, text: str) -> Optional[List[Tuple[str, float]]]:
        """Опрашивает локальную LLM (Ollama) с динамическим определением модели."""
        import urllib.request
        import urllib.error

        model_name = self._detect_ollama_model()
        available_keys = [k for k in expressions.PRESETS.keys() if k != "neutral"]

        prompt = (
            f"Ты — аниматор лица персонажа Nori. Твоя задача: проанализировать реплику пользователя '{text}' "
            f"и сопоставить ей 1-2 эмоции или направления взгляда из доступного списка: {available_keys}. "
            "Учитывай скрытый смысл, иронию, контекст, отрицания ('не злись' -> нейтраль/улыбка) и сложные действия. "
            "Для каждой выбранной эмоции укажи интенсивность от 0.1 до 1.0. "
            "Ответь ИСКЛЮЧИТЕЛЬНО валидным JSON-массивом пар вида [[\"название\", 0.8]]. Без пояснений и без markdown."
        )

        data = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_predict": 60,
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                raw_response = res.get("response", "[]").strip()

                # Очистка возможных markdown-тегов ```json ... ```
                if "```" in raw_response:
                    raw_response = re.sub(r"```(?:json)?", "", raw_response).strip()

                parsed = json.loads(raw_response)

                valid = []
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            name, weight = item
                            if name in expressions.PRESETS:
                                valid.append((str(name), float(weight)))
                elif isinstance(parsed, dict):
                    # Если модель вернула объект вида {"smile": 0.8}
                    for name, weight in parsed.items():
                        if name in expressions.PRESETS and isinstance(weight, (int, float)):
                            valid.append((str(name), float(weight)))

                if valid:
                    return valid
        except Exception:
            pass

        return None
