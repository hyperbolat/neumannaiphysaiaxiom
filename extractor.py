# -*- coding: utf-8 -*-
"""
Извлечение сущностей (какое число — какая физическая переменная) из текста —
версия на предобученном трансформере вместо BiLSTM с нуля.

Почему перешли: см. parserx.py — тот же диагноз (маленький словарь с нуля не
тянет OOV-слова и падежные формы), только здесь ещё острее, потому что
extraction — по-токенная задача, и цена ошибки на КАЖДОМ токене выше.

Как выравниваются метки с subword-токенизацией: наш data_generator.py
сохраняет посимвольные спаны {var: [start, end]} для каждой переменной.
Токенизатор трансформера (fast-токенизатор) при вызове с
return_offsets_mapping=True возвращает для каждого subword-токена его
собственные (start, end) в исходном тексте — сопоставляем спаны с
токенами по пересечению символьных диапазонов, а не по словам. Это снимает
проблему "у BERT свой word-piece словарь, не совпадающий с нашим
word-level токенайзером" в принципе — токенизация не наша забота, только
разметка поверх нее.

Число, разбитое на несколько subword-кусков (например "200000.0" может
распасться на несколько частей), помечается ОДНОЙ и той же меткой на всех
кусках; при извлечении соседние токены с одинаковой меткой склеиваются
обратно в один числовой спан перед парсингом в float.

ВАЖНО: не исполнено в песочнице (нет torch/transformers/интернета для
загрузки весов). Логика выравнивания спанов с offset_mapping — стандартный
паттерн HuggingFace для token classification, но саму torch/transformers
часть нужно прогнать и проверить на твоей машине.

Требует: pip install transformers
"""

import json
import os
import random
import re

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "cointegrated/rubert-tiny2"

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "extractor_bert.pt")
LABELS_PATH = os.path.join(WEIGHTS_DIR, "extractor_labels.json")

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_train.jsonl")

MAX_LEN = 64

# "2*10^5", "1.6×10^-19", "3·10^8" — распространённая ручная запись научной
# нотации. Нормализуем в обычное десятичное число ДО токенизации (тот же
# фикс, что уже был в предыдущей версии extractor.py — здесь он так же нужен).
SCI_NOTATION_RE = re.compile(r'(\d+(?:\.\d+)?)\s*[*×·xX]\s*10\s*\^\s*(-?\d+)')


def normalize_scientific_notation(text: str) -> str:
    def _replace(m):
        base = float(m.group(1))
        exp = int(m.group(2))
        return str(base * (10 ** exp))
    return SCI_NOTATION_RE.sub(_replace, text)


# Приставки единиц измерения ("5 мкФ", "40 см", "8 мкКл") — сознательно НЕ
# обучали модель распознавать масштаб приставки: это не задача для нейросети
# (она учит СЕМАНТИКУ, какое число что означает), а детерминированное
# правило. Правильнее выделить это в отдельный, предсказуемый шаг после
# извлечения: модель находит правильное число и правильную метку, а масштаб
# приставки домножается отдельно. Найдено на живом прогоне: "5 мкФ" (должно
# быть 5e-6) извлекалось моделью как 5.0 — приставка "мк" игнорировалась
# просто потому, что обучающий корпус НИКОГДА не использовал приставочную
# запись (всегда писал сырое десятичное число в базовой единице СИ).
UNIT_SCALE = {
    "мккл": 1e-6, "мкл": 1e-6, "нкл": 1e-9, "пкл": 1e-12,
    "мкф": 1e-6, "нф": 1e-9, "пф": 1e-12,
    "ком": 1e3,
    "мм": 1e-3, "см": 1e-2, "км": 1e3,
    "мг": 1e-6, "г": 1e-3, "т": 1e3,
    "мс": 1e-3, "мкс": 1e-6, "нс": 1e-9,
}
UNIT_WORD_RE = re.compile(r'[a-zа-яё]+', re.IGNORECASE)


def _apply_unit_scale(text: str, end_pos: int, value: float) -> float:
    """Смотрит на слово сразу после числа; если это известная приставочная
    единица — домножает значение на соответствующий масштаб. Любое
    незнакомое или базовое слово (просто "м", "кг", "с", "Н"...) не входит
    в таблицу и оставляет значение как есть — безопасный дефолт."""
    tail = text[end_pos:end_pos + 12].strip().lower()
    m = UNIT_WORD_RE.match(tail)
    if not m:
        return value
    return value * UNIT_SCALE.get(m.group(), 1.0)


class LabelSet:
    def __init__(self, label_lists=None):
        if label_lists is None:
            self.itos = ["O"]
        else:
            labels = sorted({lab for labs in label_lists for lab in labs if lab != "O"})
            self.itos = ["O"] + labels
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def to_json(self):
        return {"itos": self.itos}

    @classmethod
    def from_json(cls, data):
        v = cls.__new__(cls)
        v.itos = data["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v


class BertTagger(nn.Module):
    def __init__(self, num_labels: int, model_name: str = MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)   # (batch, seq, hidden)
        return self.fc(h)                          # (batch, seq, num_labels)


class PyTorchExtractor:
    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH, force_retrain: bool = False,
                 model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                f"Токенизатор {model_name} не 'fast' — нужен offset_mapping для "
                f"выравнивания меток по символам. Выбери модель с fast-токенизатором."
            )

        if force_retrain or not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
            self._train_and_save(dataset_path)
        else:
            self._load()

        print(f"[BERT-Extractor] Готов ({len(self.labels)} меток, база {self.model_name}) "
              f"на девайсе: {self.device}")

    # ------------------------------------------------------------------
    def _encode_with_labels(self, text: str, spans: dict, label_to_id: dict):
        """Токенизирует один пример и строит label_ids, выровненные по
        символьным спанам. -100 — служебное значение, игнорируется в loss
        (паддинг и спецтокены [CLS]/[SEP])."""
        enc = self.tokenizer(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        label_ids = []
        for start, end in offsets:
            if start == end:
                label_ids.append(-100)
                continue
            label = "O"
            for var, (vs, ve) in spans.items():
                if start >= vs and end <= ve:
                    label = var
                    break
            label_ids.append(label_to_id.get(label, 0))
        return enc["input_ids"], label_ids

    def _pad_batch(self, ids_list, label_ids_list):
        max_len = max(len(x) for x in ids_list)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for ids, labs in zip(ids_list, label_ids_list):
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention_mask.append([1] * len(ids) + [0] * pad_n)
            labels.append(labs + [-100] * pad_n)
        return (torch.tensor(input_ids, dtype=torch.long).to(self.device),
                torch.tensor(attention_mask, dtype=torch.long).to(self.device),
                torch.tensor(labels, dtype=torch.long).to(self.device))

    # ------------------------------------------------------------------
    def _train_and_save(self, dataset_path: str, epochs: int = 8, batch_size: int = 16,
                         lr: float = 2e-5, val_split: float = 0.15, seed: int = 0):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Не найден обучающий корпус: {dataset_path}. "
                f"Сгенерируй: python3 data_generator.py --per-template 200 --out dataset_train.jsonl"
            )

        random.seed(seed)
        torch.manual_seed(seed)

        rows = [json.loads(line) for line in open(dataset_path, encoding="utf-8")]
        rows = [r for r in rows if "spans" in r]
        if not rows:
            raise ValueError(
                "В корпусе нет поля 'spans' — перегенерируй датасет текущей версией "
                "data_generator.py (с посимвольными спанами для BERT-выравнивания)."
            )
        random.shuffle(rows)

        labels = LabelSet([list(r["spans"].keys()) + ["O"] for r in rows])
        label_to_id = labels.stoi

        n_val = max(1, int(len(rows) * val_split))
        train_rows, val_rows = rows[n_val:], rows[:n_val]

        def prep(rs):
            ids_list, labs_list = [], []
            for r in rs:
                spans = {k: tuple(v) for k, v in r["spans"].items()}
                ids, labs = self._encode_with_labels(r["text"], spans, label_to_id)
                ids_list.append(ids)
                labs_list.append(labs)
            return ids_list, labs_list

        val_ids_list, val_labs_list = prep(val_rows)
        val_input_ids, val_mask, val_labels = self._pad_batch(val_ids_list, val_labs_list)

        model = BertTagger(len(labels), self.model_name).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        train_ids_list, train_labs_list = prep(train_rows)

        model.train()
        for epoch in range(epochs):
            perm = list(range(len(train_rows)))
            random.shuffle(perm)
            total_loss, total_tokens = 0.0, 0
            for start in range(0, len(perm), batch_size):
                idx = perm[start:start + batch_size]
                batch_ids = [train_ids_list[i] for i in idx]
                batch_labs = [train_labs_list[i] for i in idx]
                input_ids, mask, labs = self._pad_batch(batch_ids, batch_labs)

                optimizer.zero_grad()
                logits = model(input_ids, mask)
                loss = criterion(logits.transpose(1, 2), labs)
                loss.backward()
                optimizer.step()

                n_tok = (labs != -100).sum().item()
                total_loss += loss.item() * n_tok
                total_tokens += n_tok

            model.eval()
            with torch.no_grad():
                val_logits = model(val_input_ids, val_mask)
                val_pred = val_logits.argmax(-1)
                mask_valid = val_labels != -100
                correct = ((val_pred == val_labels) & mask_valid).sum().item()
                total = mask_valid.sum().item()
            model.train()
            print(f"[BERT-Extractor] epoch {epoch + 1}/{epochs}  "
                  f"loss={total_loss / max(total_tokens,1):.4f}  "
                  f"val_token_acc={correct / max(total,1):.3f}")

        model.eval()
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(labels.to_json(), f, ensure_ascii=False)

        self.model, self.labels = model, labels

    # ------------------------------------------------------------------
    def _load(self):
        with open(LABELS_PATH, encoding="utf-8") as f:
            self.labels = LabelSet.from_json(json.load(f))
        self.model = BertTagger(len(self.labels), self.model_name).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

    # ------------------------------------------------------------------
    def extract_full(self, text: str):
        """
        Возвращает (given, target):
          - given: {имя_переменной: число} — как раньше.
          - target: имя переменной, которую текст называет как искомую
            (например "Найдите ускорение" -> target='a'), или None, если
            модель не нашла явного упоминания цели.

        Работает той же моделью и тем же набором меток, что и для given-значений:
        числовой спан с меткой — это "дано", словесный спан с меткой (не
        парсится как число) — это упоминание цели. Одна архитектура решает
        обе задачи, потому что при обучении данные размечены единообразно
        (см. ⟦...⟧-разметка в data_generator.py).
        """
        text = normalize_scientific_notation(text)
        enc = self.tokenizer(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True,
                              return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)[0]
            pred_ids = logits.argmax(-1).tolist()

        given = {}
        target_candidates = []  # (span_length, label) — на случай нескольких кандидатов
        cur_label, cur_start, cur_end = None, None, None

        def _flush():
            if cur_label is None:
                return
            substr = text[cur_start:cur_end]
            try:
                value = float(substr)
                given[cur_label] = _apply_unit_scale(text, cur_end, value)
            except ValueError:
                # Нечисловой спан с реальной меткой — упоминание цели словом,
                # а не числом ("ускорение", "силу тока", "период колебаний"...).
                target_candidates.append((cur_end - cur_start, cur_label))

        for (start, end), pid in zip(offsets, pred_ids):
            if start == end:
                label = None  # спецтокен ([CLS]/[SEP]/[PAD])
            else:
                lab = self.labels.itos[pid]
                label = None if lab == "O" else lab

            if label == cur_label and label is not None:
                cur_end = end
            else:
                _flush()
                cur_label, cur_start, cur_end = label, start, end
        _flush()

        # Если несколько кандидатов на цель — берём тот, что подкреплён
        # самым длинным спаном (обычно самый уверенный/специфичный).
        target = max(target_candidates, key=lambda x: x[0])[1] if target_candidates else None

        return given, target

    def extract(self, text: str) -> dict:
        """Обратно совместимая обёртка: только {имя_переменной: значение},
        без цели. Соседние subword-токены с одинаковой предсказанной меткой
        склеиваются в один числовой спан перед парсингом — число может
        распасться на несколько кусков токенизации, но семантически это
        одно значение."""
        given, _ = self.extract_full(text)
        return given


if __name__ == "__main__":
    extractor = PyTorchExtractor()
    samples = [
        "На тело массой 3 кг действует сила 15 Н. Найдите ускорение.",
        "Два заряда 2*10^-6 Кл и 3*10^-6 Кл на расстоянии 0.4 м. Найдите силу.",
        "Лампочка с сопротивлением 15 Ом подключена к батарейке, дающей 9 вольт.",
    ]
    for s in samples:
        given, target = extractor.extract_full(s)
        print(s)
        print(f"  дано={given}  цель={target}")