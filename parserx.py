# -*- coding: utf-8 -*-
"""
Классификатор раздела физики по тексту условия — версия на предобученном
трансформере вместо обучаемых с нуля эмбеддингов.

Почему перешли: eval_manual.py на живых (не шаблонных) формулировках показал
классификацию 77.4%, а извлечение — всего 35.5%. Разбор конкретных провалов
показал системный паттерн: как только в тексте появляются слова, которых не
было в обучающем корпусе ("лампочка", "батарейка" вместо "резистор",
"источник"), или структура предложения отличается от шаблонной — модель
теряет сигнал. Словарь в 469 слов, обученный с нуля на 4200 синтетических
примерах, физически не может знать, что "лампочка" по смыслу близка к
"резистору" — а предобученный трансформер это уже знает, потому что видел
миллиарды слов живого русского текста при предобучении.

Что даёт переход конкретно:
  - Subword-токенизация (BPE/WordPiece) решает проблему OOV-слов — незнакомое
    слово распадается на знакомые модели куски, а не превращается в <unk>.
  - Она же снимает часть проблемы с падежами ("высотой" / "с высоты") —
    словоформы одного корня делят общие подтокены.
  - Предобученные веса несут семантику, которую 4200 примеров дать не могут.

Модель по умолчанию — cointegrated/rubert-tiny2: маленькая (~30M параметров),
дообучается быстро, работает локально после первого скачивания весов
(из соображений оффлайн-работы вес не считаем "нелокальным ИИ" — скачивается
один раз и кэшируется, дальше всё работает без интернета).

ВАЖНО: этот файл не был исполнен в среде разработки (нет torch/transformers/
интернета в песочнице для загрузки предобученных весов). Логика подготовки
данных (токенизация, паддинг, сборка батчей) следует стандартному, хорошо
устоявшемуся паттерну HuggingFace fine-tuning, но саму torch/transformers
часть нужно прогнать и проверить на твоей машине.

Требует: pip install transformers (torch уже должен быть установлен).
"""

import json
import os
import random

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "cointegrated/rubert-tiny2"

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "classifier_bert.pt")
CLASSES_PATH = os.path.join(WEIGHTS_DIR, "classes.json")

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_train.jsonl")

MAX_LEN = 64  # subword-токенов обычно больше, чем слов — запас больше, чем было (32)


class BertClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state                        # (batch, seq, hidden)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pooling без паддинга
        return self.fc(self.dropout(pooled))


class PyTorchParser:
    """
    Классификатор раздела физики. При первом запуске (если весов ещё нет
    на диске) дообучает BertClassifier на dataset_train.jsonl и сохраняет
    результат; при последующих запусках просто загружает сохранённые веса.
    Публичный интерфейс (predict_category) не изменился — интеграция с
    main.py не требует правок.
    """

    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH, force_retrain: bool = False,
                 model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if force_retrain or not (os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)):
            self._train_and_save(dataset_path)
        else:
            self._load()

        print(f"[BERT-Classifier] Готов ({len(self.classes)} классов, база {self.model_name}) "
              f"на девайсе: {self.device}")

    # ------------------------------------------------------------------
    def _encode_batch(self, texts):
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN,
                              return_tensors="pt")
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    # ------------------------------------------------------------------
    def _train_and_save(self, dataset_path: str, epochs: int = 6, batch_size: int = 16,
                         lr: float = 2e-5, val_split: float = 0.15, seed: int = 0):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Не найден обучающий корпус: {dataset_path}. "
                f"Сгенерируй его: python3 data_generator.py --per-template 200 --out dataset_train.jsonl"
            )

        random.seed(seed)
        torch.manual_seed(seed)

        rows = [json.loads(line) for line in open(dataset_path, encoding="utf-8")]
        random.shuffle(rows)

        classes = sorted({r["category"] for r in rows})
        class_to_idx = {c: i for i, c in enumerate(classes)}

        n_val = max(1, int(len(rows) * val_split))
        train_rows, val_rows = rows[n_val:], rows[:n_val]

        model = BertClassifier(len(classes), self.model_name).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        val_ids, val_mask = self._encode_batch([r["text"] for r in val_rows])
        val_y = torch.tensor([class_to_idx[r["category"]] for r in val_rows], dtype=torch.long).to(self.device)

        model.train()
        for epoch in range(epochs):
            perm = list(range(len(train_rows)))
            random.shuffle(perm)
            total_loss = 0.0
            for start in range(0, len(perm), batch_size):
                idx = perm[start:start + batch_size]
                batch_rows = [train_rows[i] for i in idx]
                ids, mask = self._encode_batch([r["text"] for r in batch_rows])
                y = torch.tensor([class_to_idx[r["category"]] for r in batch_rows], dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                logits = model(ids, mask)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)

            model.eval()
            with torch.no_grad():
                val_acc = (model(val_ids, val_mask).argmax(1) == val_y).float().mean().item()
            model.train()
            print(f"[BERT-Classifier] epoch {epoch + 1}/{epochs}  "
                  f"loss={total_loss / len(train_rows):.4f}  val_acc={val_acc:.3f}")

        model.eval()
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        with open(CLASSES_PATH, "w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False)

        self.model, self.classes = model, classes

    # ------------------------------------------------------------------
    def _load(self):
        with open(CLASSES_PATH, encoding="utf-8") as f:
            self.classes = json.load(f)
        self.model = BertClassifier(len(self.classes), self.model_name).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

    # ------------------------------------------------------------------
    def predict_category(self, text: str) -> str:
        ids, mask = self._encode_batch([text])
        self.model.eval()
        with torch.no_grad():
            logits = self.model(ids, mask)
            pred_idx = logits.argmax(1).item()
        return self.classes[pred_idx]

    def predict_proba(self, text: str, top_k: int = 3):
        ids, mask = self._encode_batch([text])
        self.model.eval()
        with torch.no_grad():
            probs = torch.softmax(self.model(ids, mask), dim=1)[0]
        top = torch.topk(probs, min(top_k, len(self.classes)))
        return [(self.classes[i], float(p)) for p, i in zip(top.values, top.indices)]


if __name__ == "__main__":
    parser = PyTorchParser()
    samples = [
        "Тело массой 2 кг под действием силы 10 Н. Найдите ускорение.",
        "Лампочка с сопротивлением 15 Ом подключена к батарейке, дающей 9 вольт.",
        "Между двумя точечными зарядами 5 мкКл и 8 мкКл действует электрическая сила.",
        "Найдите период колебаний математического маятника.",
    ]
    for s in samples:
        print(f"{parser.predict_category(s):45s} <- {s}")