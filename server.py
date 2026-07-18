# -*- coding: utf-8 -*-
"""
Локальный веб-сервер для чат-интерфейса. Ничего в самом ML/физическом
пайплайне не меняет — просто оборачивает уже существующие модули
(parserx, extractor, solver, explainer) в HTTP API, чтобы веб-страница
(chat.html) могла присылать текст задачи и получать готовый отчёт.

Почему веб, а не переделка PyQt-формы в чат-виджет: с моделями (classifier
+ extractor) уже понимающими и "что дано", и "что ищем" из одного текста
(этап 6), естественный интерфейс — просто прислать условие целиком, без
отдельных полей. Веб-чат ближе к этому опыту визуально, чем desktop-форма,
и не требует переписывать PyQt-層.

Запуск:
    pip install fastapi uvicorn
    python3 server.py
Потом открыть http://127.0.0.1:8000 в браузере.

ВАЖНО: не запущено и не проверено в песочнице разработки — там нет
torch/transformers (не установить без интернета), а сервер импортирует
parserx.py/extractor.py, которые их требуют. Нужно прогнать на твоей
машине.
"""

import sympy as sp
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from parserx import PyTorchParser
from extractor import PyTorchExtractor
from solver import PhysicsSolver
from explainer import PhysicsExplainer

# Модели/солвер/объяснятель поднимаются ОДИН раз при старте сервера (как и
# раньше в main.py), а не при каждом запросе — иначе каждое сообщение в
# чате заново грузило бы веса с диска.
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] Загружаю модели...")
    _state["parser"] = PyTorchParser()
    _state["extractor"] = PyTorchExtractor()
    _state["solver"] = PhysicsSolver()
    _state["explainer"] = PhysicsExplainer()
    print("[Server] Готово, слушаю запросы.")
    yield
    _state.clear()


app = FastAPI(title="PhysAI Axiom — Chat API", lifespan=lifespan)


class SolveRequest(BaseModel):
    text: str
    target_override: Optional[str] = None  # ручное указание цели, если модель не угадала/угадала неверно


def _json_safe(value):
    """given может содержать sympy.Symbol (для абстрактных буквенных
    переменных без числового значения) — такие объекты не сериализуются
    в JSON напрямую, приводим к строке для передачи на фронтенд."""
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


@app.post("/api/solve")
def solve(req: SolveRequest):
    text = req.text.strip()
    if not text:
        return {"error": "Пустой текст задачи."}

    category = _state["parser"].predict_category(text)
    given, detected_target = _state["extractor"].extract_full(text)

    target = (req.target_override or "").strip().lower() or detected_target

    if not target:
        return {
            "error": "Не удалось определить, что нужно найти. "
                     "Уточните вопрос (например: «...найдите ускорение») "
                     "или укажите цель явно.",
            "category": category,
            "given": {k: _json_safe(v) for k, v in given.items()},
        }

    result, actual_eqs = _state["solver"].calculate(dict(given), target, category, text)
    report_html = _state["explainer"].generate_report(category, given, target, result, actual_eqs)

    return {
        "category": category,
        "given": {k: _json_safe(v) for k, v in given.items()},
        "target": target,
        "target_was_detected": detected_target == target and not req.target_override,
        "result": _json_safe(result) if not isinstance(result, (int, float)) else result,
        "report_html": report_html,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("chat.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)