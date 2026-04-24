import os
import json
import joblib
import numpy as np
from utils import clean_text, LABEL_COLS

# ── Chemins modèles ──────────────────────────────────────────────────────────
BENCHMARK_MODEL_PATH  = 'models/ovr_balanced.joblib'
BENCHMARK_TFIDF_PATH  = 'models/tfidf_vectorizer.joblib'
TEACHER_CKPT_PATH     = 'models/teacher/teacher_best.pt'
TEACHER_TOKENIZER_PATH= 'models/teacher/tokenizer'
TEACHER_THRESH_PATH   = 'models/teacher/thresholds.json'


# ════════════════════════════════════════════════════════════════════════════
# BENCHMARK — OneVsRest + LogisticRegression
# ════════════════════════════════════════════════════════════════════════════
class BenchmarkPredictor:
    def __init__(self):
        self.model    = None
        self.tfidf    = None
        self.loaded   = False
        self.error    = None

    def load(self):
        try:
            self.model  = joblib.load(BENCHMARK_MODEL_PATH)
            self.tfidf  = joblib.load(BENCHMARK_TFIDF_PATH)
            self.loaded = True
        except Exception as e:
            self.error  = str(e)
            self.loaded = False

    def predict(self, text: str) -> dict:
        if not self.loaded:
            raise RuntimeError(f'Benchmark non chargé : {self.error}')
        cleaned  = clean_text(text).lower()
        vec      = self.tfidf.transform([cleaned])
        probs    = self.model.predict_proba(vec)[0]
        return {col: float(round(p, 4)) for col, p in zip(LABEL_COLS, probs)}

    def predict_batch(self, texts: list) -> list[dict]:
        if not self.loaded:
            raise RuntimeError(f'Benchmark non chargé : {self.error}')
        cleaned = [clean_text(t).lower() for t in texts]
        vecs    = self.tfidf.transform(cleaned)
        probs   = self.model.predict_proba(vecs)
        return [
            {col: float(round(p, 4)) for col, p in zip(LABEL_COLS, row)}
            for row in probs
        ]


# ════════════════════════════════════════════════════════════════════════════
# XLM-R TEACHER — Placeholder (activé après entraînement)
# ════════════════════════════════════════════════════════════════════════════
class TeacherPredictor:
    def __init__(self):
        self.model      = None
        self.tokenizer  = None
        self.thresholds = None
        self.loaded     = False
        self.error      = None
        self.device     = None

    def load(self):
        # Vérifier que les fichiers existent
        if not os.path.exists(TEACHER_CKPT_PATH):
            self.error  = (
                'Checkpoint XLM-R introuvable. '
                'Entraîne le Teacher avec AMANE_Teacher.ipynb d\'abord, '
                f'puis place le checkpoint dans {TEACHER_CKPT_PATH}'
            )
            self.loaded = False
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            import torch.nn as nn

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(TEACHER_TOKENIZER_PATH)

            # Reconstruire l'architecture
            class TeacherModel(nn.Module):
                def __init__(self, model_name, num_labels=6, dropout=0.1):
                    super().__init__()
                    self.encoder    = AutoModel.from_pretrained(model_name)
                    hidden_size     = self.encoder.config.hidden_size
                    self.dropout    = nn.Dropout(dropout)
                    self.classifier = nn.Linear(hidden_size, num_labels)

                def mean_pool(self, token_embeds, attention_mask):
                    mask   = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
                    summed = (token_embeds * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-9)
                    return summed / counts

                def forward(self, input_ids, attention_mask):
                    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    pooled  = self.mean_pool(outputs.last_hidden_state, attention_mask)
                    pooled  = self.dropout(pooled)
                    return self.classifier(pooled)

            # Charger le checkpoint
            ckpt  = torch.load(TEACHER_CKPT_PATH, map_location=self.device)
            model = TeacherModel('xlm-roberta-base')
            model.load_state_dict(ckpt['model_state'])
            model.to(self.device)
            model.eval()
            self.model = model

            # Charger les seuils
            with open(TEACHER_THRESH_PATH) as f:
                self.thresholds = json.load(f)

            self.loaded = True

        except Exception as e:
            self.error  = str(e)
            self.loaded = False

    def predict(self, text: str) -> dict:
        if not self.loaded:
            raise RuntimeError(f'Teacher non chargé : {self.error}')
        import torch
        cleaned  = clean_text(text)
        encoding = self.tokenizer(
            cleaned, max_length=512, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids   = encoding['input_ids'].to(self.device)
        attn_mask   = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids, attn_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]
        return {col: float(round(p, 4)) for col, p in zip(LABEL_COLS, probs)}

    def predict_batch(self, texts: list) -> list[dict]:
        return [self.predict(t) for t in texts]


# ════════════════════════════════════════════════════════════════════════════
# FACTORY — chargement unique via st.cache_resource
# ════════════════════════════════════════════════════════════════════════════
def load_benchmark() -> BenchmarkPredictor:
    p = BenchmarkPredictor()
    p.load()
    return p

def load_teacher() -> TeacherPredictor:
    p = TeacherPredictor()
    p.load()
    return p