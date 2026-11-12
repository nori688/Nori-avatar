"""Гибридный семантический роутер (Semantic Vector Matcher + Negation/Compound Analyzer + LLM Client).

Обеспечивает мгновенный (<1 мс) семантический поиск по расширенной базе русскоязычных фраз,
разбирает отрицания («не злись»), составные команды («улыбнись и посмотри влево»),
поддерживает динамические последовательности анимации (таймлайны)
и кэш с возможностью подключения локальной LLM.
"""

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import expressions
from timeline import SEQUENCES, list_sequences

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_FILE = PROJECT_DIR / "models" / "local_cache.json"

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

# Маппинг фраз на имена последовательностей (таймлайнов)
SEQUENCE_CORPUS: List[Tuple[str, List[str]]] = [
    # Закатывание глаз
    ("закати глаза", ["eye_roll"]),
    ("покрути глазами", ["eye_roll"]),
    ("закатывай глаза", ["eye_roll"]),
    ("глаза в потолок", ["eye_roll"]),
    ("eye roll", ["eye_roll"]),

    # Кивок
    ("кивни", ["nod"]),
    ("кивок", ["nod"]),
    ("кивай", ["nod"]),
    ("да да", ["nod"]),
    ("согласен", ["nod"]),

    # Медленный кивок
    ("кивни медленно", ["slow_nod"]),
    ("медленный кивок", ["slow_nod"]),
    ("медленно кивни", ["slow_nod"]),

    # Покачивание головой
    ("покачай головой", ["head_shake"]),
    ("покрути головой", ["head_shake"]),
    ("нет нет", ["head_shake"]),
    ("не согласен", ["head_shake"]),
    ("качай головой", ["head_shake"]),

    # Двойное моргание
    ("моргни дважды", ["double_blink"]),
    ("двойное моргание", ["double_blink"]),
    ("моргни два раза", ["double_blink"]),

    # Медленное подмигивание
    ("подмигни медленно", ["slow_wink"]),
    ("медленное подмигивание", ["slow_wink"]),

    # Вздох
    ("вздохни", ["sigh"]),
    ("тяжелый вздох", ["sigh"]),
    ("вздохни глубоко", ["sigh"]),
    ("эх", ["sigh"]),

    # Оглядывание
    ("оглянись", ["look_around"]),
    ("оглядись", ["look_around"]),
    ("осмотрись", ["look_around"]),
    ("посмотри по сторонам", ["look_around"]),

    # Злобный прищур
    ("прищурься злобно", ["angry_squint"]),
    ("злобный прищур", ["angry_squint"]),
    ("прищурь глаза", ["angry_squint"]),

    # Смущение
    ("смутись", ["embarrassed"]),
    ("засмущайся", ["embarrassed"]),
    ("стесняйся", ["embarrassed"]),
    ("мне неловко", ["embarrassed"]),

    # Задумчивость
    ("задумайся", ["thinking"]),
    ("подумай", ["thinking"]),
    ("хмм", ["thinking"]),
    ("думаю", ["thinking"]),
    ("размышляю", ["thinking"]),

    # Радостный кивок
    ("радостно кивни", ["happy_nod"]),
    ("привет", ["happy_nod"]),
    ("приветик", ["happy_nod"]),
    ("здравствуй", ["happy_nod"]),

    # Составные последовательности
    ("закати глаза и кивни", ["eye_roll", "nod"]),
    ("закати глаза и кивни медленно", ["eye_roll", "slow_nod"]),
    ("кивни и подмигни", ["nod", "slow_wink"]),
    ("покачай головой и вздохни", ["head_shake", "sigh"]),
    ("вздохни и покачай головой", ["sigh", "head_shake"]),
    ("моргни и улыбнись", ["double_blink"]),
    ("подумай и кивни", ["thinking", "nod"]),
    ("оглянись и кивни", ["look_around", "nod"]),
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
    """Быстрый локальный разбор реплики — резерв, когда Qwen недоступна.

    Никаких сетевых вызовов: разбор занимает доли миллисекунды и никогда
    не подвешивает игровой цикл. Основное решение принимает Qwen-клиент,
    а этот класс используется как запасной вариант и источник кэша.
    """

    MIN_SCORE = 0.42
    SEQ_MIN_SCORE = 0.62
    CACHE_SCORE = 0.85

    def __init__(self):
        self.phrases = [p[0] for p in CORPUS]
        self.targets = [p[1] for p in CORPUS]

        # Символьные n-граммы (char_wb 2-4) устойчивы к окончаниям и опечаткам
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), lowercase=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.phrases)

        # Отдельный TF-IDF для фраз-последовательностей (таймлайнов)
        self.seq_phrases = [p[0] for p in SEQUENCE_CORPUS]
        self.seq_targets = [p[1] for p in SEQUENCE_CORPUS]
        self.seq_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), lowercase=True
        )
        self.seq_tfidf_matrix = self.seq_vectorizer.fit_transform(self.seq_phrases)

        self.cache: Dict[str, List[Tuple[str, float]]] = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # Кэш
    # ------------------------------------------------------------------
    def _load_cache(self):
        if not CACHE_FILE.exists():
            return
        try:
            raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            self.cache = {}
            return

        cache = {}
        for key, value in raw.items():
            if not isinstance(value, list):
                continue
            presets = []
            for item in value:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                name, weight = item
                if name in expressions.PRESETS:
                    try:
                        presets.append((str(name), float(weight)))
                    except (TypeError, ValueError):
                        continue
            if presets:
                cache[str(key).strip().lower()] = presets
        self.cache = cache

    def _save_cache(self):
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {k: [[n, round(w, 3)] for n, w in v] for k, v in self.cache.items()}
            CACHE_FILE.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Матчинг
    # ------------------------------------------------------------------
    def _best_static(self, clause: str) -> Tuple[List[Tuple[str, float]], float]:
        """Ищет лучший статический пресет для одной клаузы."""
        clause = clause.strip().lower()
        if not clause:
            return [("neutral", 1.0)], 1.0

        if clause in self.cache:
            return list(self.cache[clause]), 1.0

        is_negated = bool(re.search(r"\b(не|хватит|перестань|прекрати)\b", clause))
        cleaned = re.sub(
            r"\b(не|хватит|перестань|прекрати|пожалуйста|давай|ты)\b", "", clause
        ).strip()
        search_target = cleaned if (is_negated and cleaned) else clause

        vector = self.vectorizer.transform([search_target])
        similarities = cosine_similarity(vector, self.tfidf_matrix)[0]
        best_idx = int(np.argmax(similarities))
        score = float(similarities[best_idx])
        presets = [tuple(x) for x in self.targets[best_idx]]

        if is_negated:
            inverted: List[Tuple[str, float]] = []
            for name, _ in presets:
                if name in NEGATION_RULES:
                    inverted.extend(NEGATION_RULES[name])
                else:
                    inverted.append(("neutral", 1.0))
            presets = inverted or [("neutral", 1.0)]

        return presets, score

    def _match_sequence(self, clause: str) -> Tuple[Optional[List[str]], float]:
        """Ищет подходящую последовательность анимации по одной клаузе."""
        clause = clause.strip().lower()
        if not clause:
            return None, 0.0
        vector = self.seq_vectorizer.transform([clause])
        similarities = cosine_similarity(vector, self.seq_tfidf_matrix)[0]
        best_idx = int(np.argmax(similarities))
        return self.seq_targets[best_idx], float(similarities[best_idx])

    @classmethod
    def _prefer_sequence(cls, seq_score: float, static_score: float) -> bool:
        """Анимация побеждает, только если она заметно точнее статического пресета."""
        if seq_score >= 0.999:
            return True
        return seq_score >= cls.SEQ_MIN_SCORE and seq_score > static_score + 0.03

    # ------------------------------------------------------------------
    # Публичный разбор
    # ------------------------------------------------------------------
    def route(self, text: str):
        """Разбирает фразу целиком.

        Возвращает ``(параметры | None, описание, уверенность, последовательности | None)``.
        ``None`` в параметрах означает «позу не меняем» (чистая анимация).
        """
        norm_text = " ".join(text.strip().lower().split())
        if not norm_text:
            return None, "Пустой ввод", 1.0, None

        if norm_text in self.cache:
            presets = list(self.cache[norm_text])
            return (
                expressions.blend_presets(presets),
                "Кэш: " + ", ".join(f"{p}:{w:.2f}" for p, w in presets),
                1.0,
                None,
            )

        parts = re.split(r"[,;]|\b(и|а|затем|потом|после чего)\b", norm_text)
        clauses = [
            p.strip()
            for p in parts
            if p and p.strip() and p.strip() not in ("и", "а", "затем", "потом", "после чего")
        ]
        if not clauses:
            return None, "Пустой ввод", 1.0, None

        reactions: List[Tuple[str, float]] = []
        sequences: List[str] = []
        scores: List[float] = []

        for clause in clauses:
            statics, static_score = self._best_static(clause)
            seq_names, seq_score = self._match_sequence(clause)

            if seq_names and self._prefer_sequence(seq_score, static_score):
                sequences.extend(seq_names)
                scores.append(seq_score)
            else:
                reactions.extend(statics)
                scores.append(static_score)

        score = float(np.mean(scores)) if scores else 0.0
        params = expressions.blend_presets(reactions) if reactions else None

        # Кэшируем только цельные, уверенные и однозначные разборы
        if (
            params is not None
            and not sequences
            and len(clauses) == 1
            and score >= self.CACHE_SCORE
            and norm_text not in self.cache
        ):
            self.cache[norm_text] = [(n, w) for n, w in reactions]
            self._save_cache()

        if sequences:
            desc = "Анимация: " + ", ".join(sequences)
            if reactions:
                desc += " + " + ", ".join(f"{p}:{w:.2f}" for p, w in reactions)
        elif reactions:
            desc = "Семантика ({:.2f}): {}".format(
                score, ", ".join(f"{p}:{w:.2f}" for p, w in reactions)
            )
        else:
            desc = "Нейтрально"

        return params, desc, score, sequences or None
