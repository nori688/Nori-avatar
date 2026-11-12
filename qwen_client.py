"""Клиенты локальной и облачной LLM — «мозг» аватара Nori.

Поддерживаются два провайдера:

* **Groq** (облако, бесплатный тариф) — основной, если в ``.env`` есть
  ``GROQ_API_KEY``. Отвечает за 0.3–1 с.
* **Ollama** (локальная Qwen) — запасной, если ключа нет или Groq недоступен.

Оба работают в фоновом потоке, поэтому игровой цикл не блокируется.
Ответ валидируется: неизвестные названия реакций исправляются или
отбрасываются, сила ограничивается диапазоном 0..1, названия анимаций
сверяются с реестром SEQUENCES.
"""

from __future__ import annotations

import difflib
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import expressions
from timeline import SEQUENCES

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"

OLLAMA_HOST = os.environ.get("NORI_OLLAMA", "http://127.0.0.1:11434").rstrip("/")
GROQ_HOST = os.environ.get("NORI_GROQ_HOST", "https://api.groq.com/openai/v1").rstrip("/")

# Groq отклоняет запросы с User-Agent python-urllib (403 от Cloudflare)
USER_AGENT = "NoriAvatar/1.0 (+pygame)"

_MODEL_OVERRIDE = [
    m.strip() for m in os.environ.get("NORI_QWEN_MODEL", "").split(",") if m.strip()
]
_OLLAMA_PRIORITY = ("qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:0.5b", "qwen2.5", "qwen2")
_GROQ_PRIORITY = (
    "qwen/qwen3.8-27b",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)

SYSTEM_PROMPT = """Ты — модуль мимики 2D-персонажа Nori. По русской реплике выбери реакцию.
ЭМОЦИИ (reactions): neutral=спокойствие, smile=улыбка, joy=радость, laugh=смех, sad=грусть, angry=злость, disgust=отвращение, surprise=удивление, shock=шок, wink_left=подмигнуть левым, wink_right=подмигнуть правым, smug=ухмылка, skeptical=скепсис, bored=скука, sleepy=сонливость, look_left=взгляд влево, look_right=взгляд вправо, look_up=взгляд вверх, look_down=взгляд вниз, blink=закрыть глаза, talk=говорить.
АНИМАЦИИ (sequence): eye_roll=закатить глаза, nod=кивок, slow_nod=медленный кивок, head_shake=покачать головой нет, double_blink=моргнуть дважды, slow_wink=медленное подмигивание, sigh=вздох, look_around=оглядеться, thinking=задуматься, happy_nod=радостный кивок, embarrassed=смутиться, angry_squint=злобно прищуриться.
Формат ответа — только JSON: {"reactions":{"имя":сила},"sequence":имя_или_null}
Правила:
1) 1-3 реакции, сила 0.2-1.0.
2) Просьба о движении (кивни, покачай головой, закати глаза, вздохни, оглянись, задумайся, смутись, прищурься) — заполняй sequence.
3) Просто «подмигни» — это reactions wink_left/wink_right, а «подмигни медленно» — sequence slow_wink.
4) Отрицание («не злись», «хватит грустить») — противоположное состояние: neutral или smile.
5) Если фраза нейтральная или непонятная — {"reactions":{"neutral":1.0},"sequence":null}.
Примеры:
"улыбнись" -> {"reactions":{"smile":0.9},"sequence":null}
"мне скучно" -> {"reactions":{"bored":0.8},"sequence":null}
"ты серьезно?" -> {"reactions":{"skeptical":0.8},"sequence":null}
"кивни" -> {"reactions":{},"sequence":"nod"}
"привет" -> {"reactions":{"smile":0.8},"sequence":"happy_nod"}
"посмотри налево и улыбнись" -> {"reactions":{"smile":0.7,"look_left":0.8},"sequence":null}
"не злись" -> {"reactions":{"neutral":1.0,"smile":0.3},"sequence":null}"""

REACTION_ALIASES = {
    "anger": "angry",
    "rage": "angry",
    "mad": "angry",
    "furious": "angry",
    "happy": "smile",
    "happiness": "joy",
    "joyful": "joy",
    "fun": "joy",
    "sadness": "sad",
    "upset": "sad",
    "cry": "sad",
    "fear": "surprise",
    "scared": "shock",
    "shocked": "shock",
    "wink": "wink_left",
    "winking": "wink_left",
    "skepticism": "skeptical",
    "doubt": "skeptical",
    "smirk": "smug",
    "boredom": "bored",
    "tired": "sleepy",
    "sleep": "sleepy",
    "disgusted": "disgust",
    "speak": "talk",
    "surprised": "surprise",
    "smiling": "smile",
}


def load_dotenv(path: Optional[Path] = None):
    """Читает простой .env (KEY=VALUE), не перетирая уже заданные переменные."""
    path = path or ENV_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().replace("ё", "е").split())


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class BaseLLMClient:
    """Общая логика: фоновый поток, очередь, кэш, валидация ответа."""

    label = "LLM"
    RETRY_DELAY = 8.0

    def __init__(self, cache_path=None, enabled: bool = True):
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, dict] = {}
        self.model: Optional[str] = None
        self.checked = False
        self.last_error = ""
        self.enabled = enabled

        self._queue: "queue.Queue[str]" = queue.Queue()
        self._results: "queue.Queue[Tuple[str, Optional[dict]]]" = queue.Queue()
        self._stop = threading.Event()

        self._load_cache()
        self._thread = threading.Thread(
            target=self._worker, name=f"{self.label}-worker", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Кэш
    # ------------------------------------------------------------------
    def _load_cache(self):
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        for key, value in raw.items():
            parsed = self._validate_payload(value)
            if parsed is not None:
                self.cache[_normalize(key)] = parsed

    def _save_cache(self):
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {}
            for key, value in self.cache.items():
                payload[key] = {
                    "reactions": [[n, round(w, 3)] for n, w in value["reactions"]],
                    "sequence": value["sequence"],
                }
            self.cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def _validate_payload(self, value) -> Optional[dict]:
        if not isinstance(value, dict):
            return None
        reactions: List[Tuple[str, float]] = []
        for item in value.get("reactions") or []:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            name, weight = item
            if name not in expressions.PRESETS:
                continue
            try:
                reactions.append((str(name), _clamp01(float(weight))))
            except (TypeError, ValueError):
                continue
        sequences = [s for s in (value.get("sequence") or []) if s in SEQUENCES]
        if not reactions and not sequences:
            return None
        return {"reactions": reactions, "sequence": sequences or None}

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self.enabled and self.model is not None

    def cached(self, text: str) -> Optional[dict]:
        return self.cache.get(_normalize(text))

    def submit(self, text: str):
        """Ставит запрос в очередь, не блокируя вызывающий поток."""
        if not self.enabled:
            return
        while True:  # важна только последняя команда
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)

    def poll(self):
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Фоновый воркер
    # ------------------------------------------------------------------
    def _worker(self):
        while not self._stop.is_set():
            if self.model is None and self.enabled:
                self.model = self._detect()
                self.checked = True
                if self.model is None:
                    self._fail_pending()
                    time.sleep(self.RETRY_DELAY)
                    continue

            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                break

            self._results.put((text, self._request(text)))

    def _fail_pending(self):
        while True:
            try:
                text = self._queue.get_nowait()
            except queue.Empty:
                return
            self._results.put((text, None))

    def _detect(self) -> Optional[str]:
        raise NotImplementedError

    def _request(self, text: str) -> Optional[dict]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Разбор ответа модели
    # ------------------------------------------------------------------
    def _parse(self, content: str) -> Optional[dict]:
        if not content:
            return None
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?", "", text).strip()

        data = None
        try:
            data = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    data = None
        if not isinstance(data, dict):
            return None

        raw_reactions = data.get("reactions")
        if raw_reactions is None:
            raw_reactions = {
                k: v for k, v in data.items()
                if k not in ("sequence", "animation", "anim", "seq")
            }

        if isinstance(raw_reactions, dict):
            items = list(raw_reactions.items())
        elif isinstance(raw_reactions, list):
            items = [
                (item[0], item[1]) for item in raw_reactions
                if isinstance(item, (list, tuple)) and len(item) == 2
            ]
        else:
            items = []

        reactions: List[Tuple[str, float]] = []
        for raw_name, raw_weight in items:
            name = self._resolve_reaction(raw_name)
            if name is None:
                continue
            try:
                weight = float(str(raw_weight).replace(",", "."))
            except (TypeError, ValueError):
                weight = 0.8
            reactions.append((name, round(_clamp01(weight), 3)))

        sequences: List[str] = []
        raw_seq = data.get("sequence")
        if raw_seq:
            raw_list = raw_seq if isinstance(raw_seq, list) else [raw_seq]
            for item in raw_list:
                if not isinstance(item, str):
                    continue
                resolved = self._resolve_sequence(item)
                if resolved:
                    sequences.append(resolved)
                    continue
                reaction = self._resolve_reaction(item)
                if reaction:
                    reactions.append((reaction, 0.8))

        reactions = reactions[:3]
        if not reactions and not sequences:
            return None
        return {"reactions": reactions, "sequence": sequences or None}

    @staticmethod
    def _resolve_reaction(raw_name) -> Optional[str]:
        name = _normalize(str(raw_name)).replace(" ", "_")
        if name in expressions.PRESETS:
            return name
        if name in REACTION_ALIASES:
            return REACTION_ALIASES[name]
        match = difflib.get_close_matches(name, list(expressions.PRESETS), n=1, cutoff=0.72)
        return match[0] if match else None

    @staticmethod
    def _resolve_sequence(raw_name) -> Optional[str]:
        name = _normalize(str(raw_name)).replace(" ", "_")
        if name in SEQUENCES:
            return name
        match = difflib.get_close_matches(name, list(SEQUENCES), n=1, cutoff=0.78)
        return match[0] if match else None


class GroqClient(BaseLLMClient):
    """Облачная модель через Groq (бесплатный тариф)."""

    label = "Groq"

    def __init__(self, cache_path=None, api_key: str = "", host: str = GROQ_HOST):
        self.api_key = (api_key or "").strip()
        self.host = host
        super().__init__(cache_path, enabled=bool(self.api_key))

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": USER_AGENT,
        }

    def _detect(self) -> Optional[str]:
        if not self.api_key:
            self.last_error = "Не задан GROQ_API_KEY"
            return None
        try:
            request = urllib.request.Request(
                self.host + "/models",
                headers={"Authorization": "Bearer " + self.api_key, "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            self.last_error = "Groq недоступен: {}".format(error)
            return None

        ids = []
        for model in data.get("data", []):
            if not model.get("active", True):
                continue
            outputs = model.get("output_modalities") or ["text"]
            if "text" in outputs and "audio" not in outputs:
                ids.append(model.get("id", ""))
        ids = [i for i in ids if i]

        override = os.environ.get("NORI_GROQ_MODEL", "").strip()
        for candidate in ([override] if override else []) + list(_GROQ_PRIORITY):
            if candidate and candidate in ids:
                return candidate
        if ids:
            return ids[0]

        self.last_error = "В Groq нет доступных текстовых моделей"
        return None

    def _request(self, text: str) -> Optional[dict]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.15,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.host + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            self.last_error = f"Groq HTTP {error.code}"
            return None
        except Exception as error:
            self.last_error = str(error)
            return None

        choices = data.get("choices") or []
        content = ""
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
        parsed = self._parse(content)
        if parsed is None:
            self.last_error = "Не удалось разобрать ответ Groq"
            return None

        self.last_error = ""
        self.cache[_normalize(text)] = parsed
        self._save_cache()
        return parsed


class QwenClient(BaseLLMClient):
    """Локальная Qwen через Ollama — запасной вариант без интернета."""

    label = "Ollama"

    def _detect(self) -> Optional[str]:
        try:
            with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=2.0) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            self.last_error = "Ollama недоступен: {}".format(error)
            return None

        names = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        if not names:
            self.last_error = "В Ollama нет установленных моделей"
            return None

        for override in _MODEL_OVERRIDE:
            for name in names:
                if name == override or name.startswith(override):
                    return name

        for prefix in _OLLAMA_PRIORITY:
            for name in names:
                if name.startswith(prefix):
                    return name

        for name in names:
            if "qwen" in name.lower():
                return name

        self.last_error = "Не найдена модель Qwen"
        return None

    def _request(self, text: str) -> Optional[dict]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {"temperature": 0.15, "top_p": 0.9, "num_predict": 64},
        }
        request = urllib.request.Request(
            OLLAMA_HOST + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            self.last_error = str(error)
            return None

        message = data.get("message") or {}
        parsed = self._parse(message.get("content", ""))
        if parsed is None:
            self.last_error = "Не удалось разобрать ответ Qwen"
            return None

        self.last_error = ""
        self.cache[_normalize(text)] = parsed
        self._save_cache()
        return parsed


def create_client(cache_path=None, provider: Optional[str] = None) -> BaseLLMClient:
    """Выбирает провайдера: Groq, если есть ключ, иначе локальную Ollama."""
    load_dotenv()
    prefer = (provider or os.environ.get("NORI_LLM_PROVIDER", "auto")).strip().lower()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if prefer in ("auto", "groq") and api_key:
        return GroqClient(cache_path, api_key)
    return QwenClient(cache_path)
