# -*- coding: utf-8 -*-
"""
Извлечение сущностей (какое число — какая физическая переменная) из текста
условия задачи. Заменяет цепочку regex/ключевых слов в main.py::auto_extract_data
на токен-классификатор.

Идея разметки: каждое число в сгенерированном тексте (data_generator.py)
попадает в текст ровно в одном месте — мы знаем точные символьные позиции
благодаря render_with_spans(). Токенизация даёт числа отдельными токенами
(TOKEN_RE ловит и слова, и числа), поэтому разметка получается плоской:
каждому токену — метка "O" (не число) или имя переменной ('m', 'f_lens',
'q1_ch', ...). BIO-теги (Begin/Inside) не нужны, потому что число — всегда
один токен целиком, вложенных сущностей нет.

Модель: nn.Embedding -> двунаправленный LSTM (важен контекст: "17 кг" и
"17 Н" отличаются только соседним словом) -> линейный классификатор на
каждый таймстеп.

ВАЖНО (как и с parserx.py): эта часть не была исполнена в песочнице
разработки — нет torch/интернета. Логика токенизации/выравнивания/паддинга
проверена отдельно на чистом Python (см. историю чата), но саму torch-часть
нужно прогнать на твоей машине.
"""

import json
import os
import re
import random

import torch
import torch.nn as nn
import torch.optim as optim

TOKEN_RE = re.compile(r'\d+\.?\d*(?:[eE][-+]?\d+)?|[a-zA-Zа-яА-ЯёЁ]+')
NUM_RE = re.compile(r'^\d+\.?\d*(?:[eE][-+]?\d+)?$')

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "extractor.pt")
VOCAB_PATH = os.path.join(WEIGHTS_DIR, "extractor_vocab.json")
LABELS_PATH = os.path.join(WEIGHTS_DIR, "extractor_labels.json")

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_train.jsonl")

MAX_LEN = 32
EMBED_DIM = 48
HIDDEN_DIM = 64
NUM_TOKEN = "<num>"


def tokenize(text: str):
    return TOKEN_RE.findall(text)


def normalize(tok: str) -> str:
    """Числа сворачиваются в один общий псевдо-токен <num> — конкретное
    значение (17.03 против 81.39) не несёт лексической информации для
    классификации, важен только факт "здесь стоит число" и контекст вокруг."""
    return NUM_TOKEN if NUM_RE.match(tok) else tok.lower()


class Vocab:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, token_lists=None, min_freq: int = 1):
        if token_lists is None:
            self.itos = [self.PAD, self.UNK, NUM_TOKEN]
        else:
            freq = {}
            for tokens in token_lists:
                for tok in tokens:
                    norm = normalize(tok)
                    freq[norm] = freq.get(norm, 0) + 1
            words = sorted(w for w, c in freq.items() if c >= min_freq and w != NUM_TOKEN)
            self.itos = [self.PAD, self.UNK, NUM_TOKEN] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len: int = MAX_LEN):
        ids = [self.stoi.get(normalize(t), 1) for t in tokens][:max_len]
        ids += [0] * (max_len - len(ids))
        return ids

    def to_json(self):
        return {"itos": self.itos}

    @classmethod
    def from_json(cls, data):
        v = cls.__new__(cls)
        v.itos = data["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v


class LabelSet:
    """Множество меток: 'O' + все имена переменных, встреченные в корпусе."""

    def __init__(self, label_lists=None):
        if label_lists is None:
            self.itos = ["O"]
        else:
            labels = sorted({lab for labs in label_lists for lab in labs if lab != "O"})
            self.itos = ["O"] + labels
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, labels, max_len: int = MAX_LEN):
        ids = [self.stoi.get(l, 0) for l in labels][:max_len]
        ids += [0] * (max_len - len(ids))  # паддинг размечаем как "O", но он всё равно маскируется в loss/inference
        return ids

    def to_json(self):
        return {"itos": self.itos}

    @classmethod
    def from_json(cls, data):
        v = cls.__new__(cls)
        v.itos = data["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v


class TaggerModel(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int,
                 embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)                 # (batch, seq_len, embed_dim)
        out, _ = self.lstm(emb)                  # (batch, seq_len, hidden_dim*2)
        out = self.dropout(out)
        return self.fc(out)                       # (batch, seq_len, num_labels)


class PyTorchExtractor:
    """
    Извлекает {переменная: значение} из текста условия. Как и PyTorchParser:
    обучается один раз (если весов ещё нет на диске), дальше просто грузит.
    """

    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH, force_retrain: bool = False):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        if force_retrain or not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)
                                  and os.path.exists(LABELS_PATH)):
            self._train_and_save(dataset_path)
        else:
            self._load()

        print(f"[Extractor] Готов ({len(self.labels)} меток переменных, "
              f"словарь {len(self.vocab)} токенов) на девайсе: {self.device}")

    # ------------------------------------------------------------------
    def _train_and_save(self, dataset_path: str, epochs: int = 25, batch_size: int = 32,
                         lr: float = 1e-3, val_split: float = 0.15, seed: int = 0):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Не найден обучающий корпус: {dataset_path}. "
                f"Сгенерируй его: python3 data_generator.py --per-template 150 --out dataset_train.jsonl"
            )

        random.seed(seed)
        torch.manual_seed(seed)

        rows = [json.loads(line) for line in open(dataset_path, encoding="utf-8")]
        rows = [r for r in rows if "tokens" in r and "labels" in r]
        if not rows:
            raise ValueError(
                "В корпусе нет полей 'tokens'/'labels' — перегенерируй датасет "
                "текущей версией data_generator.py (со span-разметкой)."
            )
        random.shuffle(rows)

        vocab = Vocab([r["tokens"] for r in rows], min_freq=1)
        labels = LabelSet([r["labels"] for r in rows])

        n_val = max(1, int(len(rows) * val_split))
        train_rows, val_rows = rows[n_val:], rows[:n_val]

        def make_batch(row_slice):
            X = torch.tensor([vocab.encode(r["tokens"]) for r in row_slice], dtype=torch.long)
            Y = torch.tensor([labels.encode(r["labels"]) for r in row_slice], dtype=torch.long)
            mask = torch.tensor(
                [[1 if i < len(r["tokens"]) else 0 for i in range(MAX_LEN)] for r in row_slice],
                dtype=torch.bool,
            )
            return X, Y, mask

        model = TaggerModel(len(vocab), len(labels)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(reduction="none")

        X_val, Y_val, M_val = make_batch(val_rows)
        X_val, Y_val, M_val = X_val.to(self.device), Y_val.to(self.device), M_val.to(self.device)

        model.train()
        for epoch in range(epochs):
            perm = list(range(len(train_rows)))
            random.shuffle(perm)
            total_loss, total_tokens = 0.0, 0
            for start in range(0, len(perm), batch_size):
                idx = perm[start:start + batch_size]
                batch_rows = [train_rows[i] for i in idx]
                X, Y, mask = make_batch(batch_rows)
                X, Y, mask = X.to(self.device), Y.to(self.device), mask.to(self.device)

                optimizer.zero_grad()
                logits = model(X)                                    # (batch, seq_len, num_labels)
                loss_per_tok = criterion(logits.transpose(1, 2), Y)   # (batch, seq_len)
                loss = (loss_per_tok * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val)
                    val_pred = val_logits.argmax(-1)
                    correct = ((val_pred == Y_val) & M_val).sum().item()
                    total = M_val.sum().item()
                model.train()
                print(f"[Extractor] epoch {epoch + 1}/{epochs}  "
                      f"loss={total_loss / max(total_tokens,1):.4f}  "
                      f"val_token_acc={correct / max(total,1):.3f}")

        model.eval()
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab.to_json(), f, ensure_ascii=False)
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(labels.to_json(), f, ensure_ascii=False)

        self.model, self.vocab, self.labels = model, vocab, labels

    # ------------------------------------------------------------------
    def _load(self):
        with open(VOCAB_PATH, encoding="utf-8") as f:
            self.vocab = Vocab.from_json(json.load(f))
        with open(LABELS_PATH, encoding="utf-8") as f:
            self.labels = LabelSet.from_json(json.load(f))

        self.model = TaggerModel(len(self.vocab), len(self.labels)).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

    # ------------------------------------------------------------------
    def extract(self, text: str) -> dict:
        """
        Возвращает {имя_переменной: числовое_значение}, разобранные из текста.
        Если одна переменная встречается несколько раз (не должно происходить
        в норме), побеждает последнее по тексту вхождение.
        """
        tokens = tokenize(text)
        ids = torch.tensor([self.vocab.encode(tokens)], dtype=torch.long).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(ids)[0]                     # (seq_len, num_labels)
            pred_ids = logits.argmax(-1).tolist()

        result = {}
        for i, tok in enumerate(tokens[:MAX_LEN]):
            label = self.labels.itos[pred_ids[i]]
            if label == "O":
                continue
            if not NUM_RE.match(tok):
                continue  # модель предсказала переменную не на числовом токене — считаем шумом
            try:
                result[label] = float(tok)
            except ValueError:
                continue
        return result


if __name__ == "__main__":
    extractor = PyTorchExtractor()
    samples = [
        "На тело массой 3 кг действует сила 15 Н. Найдите ускорение.",
        "Два заряда 2e-06 Кл и 3e-06 Кл на расстоянии 0.4 м. Найдите силу.",
        "Резистор сопротивлением 10 Ом подключен к источнику напряжением 20 В.",
    ]
    for s in samples:
        print(s)
        print("  ->", extractor.extract(s))