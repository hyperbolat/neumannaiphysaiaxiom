# -*- coding: utf-8 -*-
"""
Классификатор раздела физики по тексту условия.

Отличия от предыдущей версии (важно понимать, что изменилось и почему):

  1. Раньше словарь был вручную подобранным списком из ~48 слов, а
     "векторизация" — one-hot счётчиком совпадений подстрок. Теперь словарь
     строится автоматически из обучающего корпуса (data_generator.py), 
     токенизация — честная, по регулярке на русские/латинские буквы.

  2. Раньше сеть обучалась заново при КАЖДОМ запуске на 11 примерах — это
     не обучение, а по сути memoization. Теперь: обучение запускается один
     раз на ~3000+ синтетических примерах с train/val-разбиением, веса и
     словарь сохраняются на диск (weights/classifier.pt, weights/vocab.json)
     и при следующих запусках просто загружаются.

  3. Раньше эмбеддингов не было вообще — только счётчики. Теперь у каждого
     слова есть обучаемый вектор (nn.Embedding), предложение представляется
     как усреднение векторов его слов (mean pooling) — это разумный локальный
     аналог без необходимости скачивать предобученные модели (в песочнице
     разработки нет интернета для sentence-transformers; при желании эту
     nn.Embedding-прослойку можно позже заменить на предобученные эмбеддинги
     без изменения остального пайплайна — интерфейс тот же).

  4. Класс PyTorchParser сохранил тот же публичный интерфейс
     (predict_category), поэтому интеграция с main.py не требует изменений
     в вызывающем коде.

ВАЖНО: этот файл не был исполнен в среде разработки (нет torch/интернета
в песочнице) — логика токенизации и словаря проверена отдельно на чистом
Python, но саму торчовую часть нужно прогнать на твоей машине.
"""

import json
import os
import re
import random

import torch
import torch.nn as nn
import torch.optim as optim

TOKEN_RE = re.compile(r'[a-zA-Zа-яА-ЯёЁ]+')

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "classifier.pt")
VOCAB_PATH = os.path.join(WEIGHTS_DIR, "vocab.json")
CLASSES_PATH = os.path.join(WEIGHTS_DIR, "classes.json")

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_train.jsonl")

MAX_LEN = 32
EMBED_DIM = 64
HIDDEN_DIM = 64


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


class Vocab:
    """Словарь токен<->индекс, построенный по частоте слов в корпусе."""

    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, token_lists=None, min_freq: int = 1):
        if token_lists is None:
            self.itos = [self.PAD, self.UNK]
        else:
            freq = {}
            for tokens in token_lists:
                for tok in tokens:
                    freq[tok] = freq.get(tok, 0) + 1
            vocab_words = sorted(w for w, c in freq.items() if c >= min_freq)
            self.itos = [self.PAD, self.UNK] + vocab_words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len: int = MAX_LEN):
        ids = [self.stoi.get(t, 1) for t in tokens][:max_len]
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


class EmbedClassifier(nn.Module):
    """Усреднённые обучаемые эмбеддинги слов -> MLP-голова классификации."""

    def __init__(self, vocab_size: int, num_classes: int,
                 embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) индексы токенов, 0 = паддинг
        emb = self.embedding(x)                                   # (batch, seq_len, embed_dim)
        mask = (x != 0).unsqueeze(-1).float()                      # (batch, seq_len, 1)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # mean pooling без учёта паддинга
        h = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(h)


class PyTorchParser:
    """
    Классификатор раздела физики. При первом запуске (если весов ещё нет
    на диске) обучается на dataset_train.jsonl и сохраняет результат;
    при последующих запусках просто загружает сохранённые веса и словарь.
    """

    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH, force_retrain: bool = False):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        if force_retrain or not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)
                                  and os.path.exists(CLASSES_PATH)):
            self._train_and_save(dataset_path)
        else:
            self._load()

        print(f"[PyTorch] Классификатор готов ({len(self.classes)} классов, "
              f"словарь {len(self.vocab)} слов) на девайсе: {self.device}")

    # ------------------------------------------------------------------
    def _train_and_save(self, dataset_path: str, epochs: int = 20, batch_size: int = 32,
                         lr: float = 1e-3, val_split: float = 0.15, seed: int = 0):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Не найден обучающий корпус: {dataset_path}. "
                f"Сгенерируй его: python3 data_generator.py --per-template 150 --out dataset_train.jsonl"
            )

        random.seed(seed)
        torch.manual_seed(seed)

        rows = [json.loads(line) for line in open(dataset_path, encoding="utf-8")]
        random.shuffle(rows)

        classes = sorted({r["category"] for r in rows})
        class_to_idx = {c: i for i, c in enumerate(classes)}

        tokenized = [tokenize(r["text"]) for r in rows]
        vocab = Vocab(tokenized, min_freq=1)

        n_val = max(1, int(len(rows) * val_split))
        train_rows, val_rows = rows[n_val:], rows[:n_val]
        train_tok, val_tok = tokenized[n_val:], tokenized[:n_val]

        def make_batch(row_slice, tok_slice):
            X = torch.tensor([vocab.encode(t) for t in tok_slice], dtype=torch.long)
            y = torch.tensor([class_to_idx[r["category"]] for r in row_slice], dtype=torch.long)
            return X, y

        model = EmbedClassifier(len(vocab), len(classes)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_val, y_val = make_batch(val_rows, val_tok)
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)

        model.train()
        for epoch in range(epochs):
            perm = list(range(len(train_rows)))
            random.shuffle(perm)
            total_loss = 0.0
            for start in range(0, len(perm), batch_size):
                idx = perm[start:start + batch_size]
                batch_rows = [train_rows[i] for i in idx]
                batch_tok = [train_tok[i] for i in idx]
                X, y = make_batch(batch_rows, batch_tok)
                X, y = X.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()
                model.train()
                print(f"[PyTorch] epoch {epoch + 1}/{epochs}  "
                      f"loss={total_loss / len(train_rows):.4f}  val_acc={val_acc:.3f}")

        model.eval()
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab.to_json(), f, ensure_ascii=False)
        with open(CLASSES_PATH, "w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False)

        self.model, self.vocab, self.classes = model, vocab, classes

    # ------------------------------------------------------------------
    def _load(self):
        with open(VOCAB_PATH, encoding="utf-8") as f:
            self.vocab = Vocab.from_json(json.load(f))
        with open(CLASSES_PATH, encoding="utf-8") as f:
            self.classes = json.load(f)

        self.model = EmbedClassifier(len(self.vocab), len(self.classes)).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

    # ------------------------------------------------------------------
    def predict_category(self, text: str) -> str:
        tokens = tokenize(text)
        ids = torch.tensor([self.vocab.encode(tokens)], dtype=torch.long).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(ids)
            pred_idx = logits.argmax(1).item()
        return self.classes[pred_idx]

    def predict_proba(self, text: str, top_k: int = 3):
        """Полезно для отладки/UI: топ-k предсказаний с вероятностями."""
        tokens = tokenize(text)
        ids = torch.tensor([self.vocab.encode(tokens)], dtype=torch.long).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probs = torch.softmax(self.model(ids), dim=1)[0]
        top = torch.topk(probs, min(top_k, len(self.classes)))
        return [(self.classes[i], float(p)) for p, i in zip(top.values, top.indices)]


if __name__ == "__main__":
    # Быстрая проверка: обучить (или загрузить) и прогнать пару примеров.
    parser = PyTorchParser()
    samples = [
        "Тело массой 2 кг под действием силы 10 Н. Найдите ускорение.",
        "Два точечных заряда взаимодействуют на расстоянии. Найдите силу.",
        "Найдите период колебаний математического маятника.",
        "Резистор сопротивлением подключен к источнику напряжения. Найдите ток.",
    ]
    for s in samples:
        print(f"{parser.predict_category(s):45s} <- {s}")