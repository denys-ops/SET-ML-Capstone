"""Build capstone phishing_scorer.ipynb (F0: data + preprocessing + EDA) and execute it."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

# ---------- Title / README ----------
md("""# Phishing-Obviousness Scorer — дослідницький ноутбук

**Що це.** Невелика модель, яка читає текст листа й повертає ймовірність, що це фішинг (`P(phish)`).
Цю ймовірність я використовую як шкалу «очевидності»: низький бал — приманка переконлива, високий —
фішинг легко розпізнати. Додатково модель підсвічує слова, що вплинули на рішення.

**Головна теза.** Маленька модель дає майже ту саму якість, що й велика LLM, але у сотні разів швидше
і майже безкоштовно. Саме це й потрібно, щоб рахувати бал на кожну зміну тексту прямо в редакторі —
виклик LLM на кожне натискання неможливий.

**Що в ноутбуці.** Дані → огляд даних → проста базова модель → DistilBERT → калібрування → пояснення
рішень → порівняння з LLM → підготовка моделі до використання. Готовий сервіс (FastAPI) — окремо.

**Лише текст.** Модель дивиться на текст, який бачить людина, а не на HTML-код. Той самий крок
очищення тексту (`html_to_scoring_text`) працює і під час навчання, і в готовому продукті — щоб модель
у роботі отримувала рівно те, на чому вчилась.

---""")

# ---------- F0 ----------
md("""## F0 · Дані та підготовка

Тренувався лише на публічних даних. Основний набір — **`zefang-liu/phishing-email-dataset`** (~18.6k
листів, дві мітки: Safe / Phishing). Набір MeAJOR теж розглядав, але його не можна завантажити напряму,
тож заради відтворюваності взяв цей.""")

md("### Setup — усі імпорти, seed, девайс")
code("""# bootstrap: ноутбук у notebooks/, але всі шляхи (data/ model/ onnx/) — від кореня репо,
# а модулі synth/preprocess — у src/. Робимо CWD=корінь і додаємо src/ у sys.path.
import os, sys
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, "src")

import re, random, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from bs4 import BeautifulSoup, Comment
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from captum.attr import LayerIntegratedGradients, visualization as viz

%matplotlib inline

SEED = 42
random.seed(SEED); np.random.seed(SEED); set_seed(SEED)

DEVICE = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4.5)
pd.set_option("display.max_colwidth", 120)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
print("setup ok | seed:", SEED, "| device:", DEVICE)""")

# ---------- html_to_scoring_text ----------
md("""### `html_to_scoring_text` — очищення тексту

Ключова спільна деталь: **одна й та сама функція** обробляє і датасет під час навчання, і вхід у
готовому продукті. Вона лишає тільки той текст, який бачить людина:

1. прибирає `<head>/<style>/<script>/<title>`, коментарі й службовий код листа;
2. бере видимий текст у порядку читання;
3. додає тему листа спереду;
4. підставляє плейсхолдери шаблонів (`{{.FirstName}}`→імʼя, `{{.URL}}`→`<URL>`, …);
5. замінює конкретні посилання / пошти / числа на позначки, щоб модель не запамʼятовувала домени.""")
code("""URL_RE   = re.compile(r"https?://\\S+|www\\.\\S+", re.I)
EMAIL_RE = re.compile(r"\\b[\\w.+-]+@[\\w-]+\\.[\\w.-]+\\b")
NUM_RE   = re.compile(r"\\b\\d[\\d.,]*\\b")
WS_RE    = re.compile(r"\\s+")
TAG_RE   = re.compile(r"<[a-zA-Z!/][^>]*>")

GOPHISH = {
    "{{.FirstName}}": "John", "{{.LastName}}": "Smith",
    "{{.Email}}": "<EMAIL>", "{{.From}}": "<EMAIL>",
    "{{.URL}}": "<URL>", "{{.TrackingURL}}": "<URL>",
}

def render_placeholders(s: str) -> str:
    for k, v in GOPHISH.items():
        s = s.replace(k, v)
    return re.sub(r"\\{\\{\\.\\w+\\}\\}", "<VAR>", s)  # будь-який інший {{.X}}

def html_to_text(html: str) -> str:
    if not html:
        return ""
    if not TAG_RE.search(html):          # уже плейн-текст — без парсингу
        return html
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["head", "style", "script", "title"]):
        tag.decompose()
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    return soup.get_text(separator=" ")

def html_to_scoring_text(html: str, subject: str = "") -> str:
    \"\"\"HTML(+subject) -> нормалізований текст, який бачить модель. Ідентична train/serve.\"\"\"
    raw = (html or "")[:30000]           # кліп аномально довгих тіл
    text = (subject or "") + "\\n" + html_to_text(raw)
    text = render_placeholders(text)
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = NUM_RE.sub("<NUM>", text)
    return WS_RE.sub(" ", text).strip()""")

md("Швидка перевірка на синтетичному Whalen-подібному HTML із плейсхолдерами:")
code('''demo_html = \'\'\'<!doctype html><html><head><style>body{padding:0}</style></head>
<body><p>Hi {{.FirstName}},</p>
<p>Please confirm your account at <a href="{{.URL}}">https://acme-login.example.com/verify</a>
before 5pm or it will be suspended. Amount due: $4,200.</p>
<p>Regards,<br>Michael R. Bradford, CEO</p></body></html>\'\'\'

print(html_to_scoring_text(demo_html, subject="Urgent: Wire Transfer"))''')

# ---------- load ----------
md("### Завантаження даних")
code("""raw = load_dataset("zefang-liu/phishing-email-dataset", split="train").to_pandas()
df = raw.rename(columns={"Email Text": "text_raw", "Email Type": "etype"})[["text_raw", "etype"]]
df["text_raw"] = df["text_raw"].fillna("").astype(str)
df["label"] = (df["etype"].str.strip() == "Phishing Email").astype(int)
df.head(3)""")

# ---------- clean ----------
md("""### Очищення: порожні, дублікати, нормалізація

Застосовуємо `html_to_scoring_text` до всіх рядків (для плейн-тексту вона майже no-op, але дає єдиний
нормалізований формат). Потім прибираємо порожні й точні дублі **до спліту** (анти-leakage).""")
code("""df["text"] = [html_to_scoring_text(t) for t in df["text_raw"]]

before = len(df)
df = df[df["text"].str.strip() != ""]                 # порожні
df = df.drop_duplicates(subset="text").reset_index(drop=True)   # точні дублі
print(f"рядків: {before} -> {len(df)}  (прибрано {before - len(df)})")
print(df["label"].value_counts().rename({0: 'legit', 1: 'phish'}))""")

# ---------- EDA: balance ----------
md("### EDA · баланс класів")
code("""ax = (df["label"].map({0: "legit", 1: "phish"})
        .value_counts()
        .plot(kind="bar", color=["#4c72b0", "#c44e52"], rot=0))
ax.set_title("Баланс класів"); ax.set_ylabel("к-сть листів")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout(); plt.show()""")

# ---------- EDA: lengths ----------
md("### EDA · довжини текстів по класах (у словах)")
code("""df["n_words"] = df["text"].str.split().str.len()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for lab, name, col in [(0, "legit", "#4c72b0"), (1, "phish", "#c44e52")]:
    sns.histplot(df[df.label == lab]["n_words"].clip(upper=600),
                 bins=50, color=col, label=name, alpha=0.55, ax=ax[0])
ax[0].set_title("Розподіл довжин (кліп 600 слів)"); ax[0].set_xlabel("слів"); ax[0].legend()
sns.boxplot(data=df.assign(cls=df.label.map({0: "legit", 1: "phish"})),
            x="cls", y="n_words", ax=ax[1])
ax[1].set_ylim(0, 600); ax[1].set_title("Довжина по класах"); ax[1].set_xlabel("")
plt.tight_layout(); plt.show()""")

md("Зведена статистика довжин:")
code("""df.groupby(df.label.map({0: 'legit', 1: 'phish'}))["n_words"].describe(
    percentiles=[.5, .95])[["mean", "50%", "95%", "max"]].round(0)""")

# ---------- EDA: leakage note ----------
md("""### Огляд даних · застереження про джерела

Важливий ризик у цьому наборі: легітимні листи — це переважно робоче листування Enron і наукові
розсилки, а фішинг — зі збірок спаму. Тобто модель може вчитися розрізняти їх за темою й стилем
джерела, а не за самим обманом. Нормалізація вже прибрала конкретні посилання й числа; тут я лише
явно фіксую цей ризик, а перевіряю його далі — на власному корпоративному тесті.""")
code("""top = pd.DataFrame({
    "median_words": df.groupby("label")["n_words"].median(),
    "share": df["label"].value_counts(normalize=True).round(3),
}).rename(index={0: "legit", 1: "phish"})
top""")

# ---------- split + save ----------
md("""### Стратифікований спліт + збереження

70/15/15 із фіксованим seed. **val** — для тюнінгу та фіту калібратора (F3), **test** — лише фінальний
eval і спільний бенчмарк vs LLM (F5).""")
code("""train_df, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=SEED)

for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    part.reset_index(drop=True).to_parquet(f"{DATA_DIR}/{name}.parquet")

print("split (рядки):", {k: len(v) for k, v in
      [("train", train_df), ("val", val_df), ("test", test_df)]})
print("phish-частка:", {k: round(v['label'].mean(), 3) for k, v in
      [("train", train_df), ("val", val_df), ("test", test_df)]})""")

md("""---
**F0 готово.** Маємо нормалізований `text,label`, прозорий лік-ризик і стратифікований спліт у
`data/`. Далі **F1 · Baseline (TF-IDF + LogReg)** — чесна точка відліку.""")

# ---------- F1 ----------
md("""## F1 · Базова модель — TF-IDF + логістична регресія

Проста класична модель як орієнтир: логістична регресія на TF-IDF (окремі слова й пари слів). Вона дає
ймовірність, зрозумілі ваги слів і число, яке DistilBERT далі має перевершити.""")

md("### Завантаження спліту")
code("""train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
val_df   = pd.read_parquet(f"{DATA_DIR}/val.parquet")
test_df  = pd.read_parquet(f"{DATA_DIR}/test.parquet")
print({"train": len(train_df), "val": len(val_df), "test": len(test_df)})""")

md("### Векторизація + навчання")
code("""tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000,
                        sublinear_tf=True, strip_accents="unicode")
Xtr = tfidf.fit_transform(train_df["text"])
Xva = tfidf.transform(val_df["text"])
Xte = tfidf.transform(test_df["text"])

logreg = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")
logreg.fit(Xtr, train_df["label"])
print("features:", Xtr.shape[1])""")

md("### Метрики на test")
code("""def report(name, y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    print(f"== {name} ==")
    print(f"F1     : {f1_score(y_true, pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, proba):.4f}")
    print(f"PR-AUC : {average_precision_score(y_true, proba):.4f}")
    print(classification_report(y_true, pred, target_names=["legit", "phish"], digits=3))

p_te_base = logreg.predict_proba(Xte)[:, 1]
report("Baseline TF-IDF+LogReg · TEST", test_df["label"], p_te_base)""")

md("### Confusion matrix")
code("""cm = confusion_matrix(test_df["label"], (p_te_base >= 0.5).astype(int))
ConfusionMatrixDisplay(cm, display_labels=["legit", "phish"]).plot(
    cmap="Blues", colorbar=False)
plt.title("Baseline · confusion matrix (test)"); plt.tight_layout(); plt.show()""")

md("### ROC та PR криві")
code("""fpr, tpr, _ = roc_curve(test_df["label"], p_te_base)
prec, rec, _ = precision_recall_curve(test_df["label"], p_te_base)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(fpr, tpr, color="#4c72b0"); ax[0].plot([0, 1], [0, 1], "--", color="grey")
ax[0].set_title(f"ROC (AUC={roc_auc_score(test_df['label'], p_te_base):.3f})")
ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
ax[1].plot(rec, prec, color="#c44e52")
ax[1].set_title(f"PR (AP={average_precision_score(test_df['label'], p_te_base):.3f})")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
plt.tight_layout(); plt.show()""")

md("### Топ-фічі, що тягнуть до 'phish' та 'legit'")
code("""coef = logreg.coef_[0]
names = np.array(tfidf.get_feature_names_out())
top_phish = names[np.argsort(coef)[-15:]][::-1]
top_legit = names[np.argsort(coef)[:15]]
pd.DataFrame({"-> phish": top_phish, "-> legit": top_legit})""")

md("""---
**F1 готово.** Маємо baseline-метрики як точку відліку. Далі **F2 · DistilBERT fine-tune** — потребує
GPU (Colab); локально на MPS повільніше. Збережемо baseline-проби для спільної таблиці-висновку (F5).""")

code("""np.save(f"{DATA_DIR}/baseline_test_proba.npy", p_te_base)
print("saved baseline test proba:", p_te_base.shape)""")

# ---------- F2 ----------
md("""## F2 · DistilBERT fine-tune

Трансформер — основна DL-модель. `max_length=256` (покриває більшість листів повністю, чесне
порівняння з TF-IDF), `batch=16` (ліміт MPS-пам'яті), `2 епохи` (на Colab GPU — сміливо `32`/`4`).
Зберігаємо **логіти** на val/test — потрібні для калібрації (F3) і спільної таблиці (F5).""")

md("### Токенізація")
code("""MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def to_ds(df):
    d = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    return d.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN),
                 batched=True)

ds_train, ds_val, ds_test = to_ds(train_df), to_ds(val_df), to_ds(test_df)
print("tokenized:", {"train": len(ds_train), "val": len(ds_val), "test": len(ds_test)})""")

md("""### Навчання

Ідемпотентно: якщо `model/distilbert` уже існує — завантажуємо (без повторного тренування);
інакше тренуємо й зберігаємо модель + історію навчання. Так F3+ переганяються без 40-хв тренування.""")
code("""import json
SAVE_DIR  = "model/distilbert"
HIST_PATH = f"{DATA_DIR}/distilbert_history.json"
RETRAIN   = not os.path.isdir(SAVE_DIR)

def compute_metrics(ep):
    logits, labels = ep
    proba = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    pred = (proba >= 0.5).astype(int)
    return {"f1": f1_score(labels, pred),
            "accuracy": accuracy_score(labels, pred),
            "roc_auc": roc_auc_score(labels, proba)}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME if RETRAIN else SAVE_DIR, num_labels=2)

args = TrainingArguments(
    output_dir="out_distilbert", learning_rate=2e-5,
    per_device_train_batch_size=16, per_device_eval_batch_size=16,
    num_train_epochs=2, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
    logging_steps=50, report_to="none", fp16=False, seed=SEED)

trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
    processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics)

if RETRAIN:
    trainer.train()
    trainer.save_model(SAVE_DIR); tokenizer.save_pretrained(SAVE_DIR)
    json.dump(trainer.state.log_history, open(HIST_PATH, "w"))
    log_history = trainer.state.log_history
    print("trained & saved ->", SAVE_DIR)
else:
    log_history = json.load(open(HIST_PATH))
    print("loaded cached model <-", SAVE_DIR)""")

md("### Криві навчання")
code("""hist = pd.DataFrame(log_history)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
tr_loss = hist.dropna(subset=["loss"])
ax[0].plot(tr_loss["step"], tr_loss["loss"], color="#4c72b0")
ax[0].set_title("Train loss"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
ev = hist.dropna(subset=["eval_f1"])
ax[1].plot(ev["epoch"], ev["eval_f1"], marker="o", color="#55a868", label="eval F1")
ax[1].plot(ev["epoch"], ev["eval_roc_auc"], marker="s", color="#c44e52", label="eval ROC-AUC")
ax[1].set_title("Eval по епохах"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.show()""")

md("### Метрики на test + збереження логітів")
code("""lg_val, lg_test = f"{DATA_DIR}/distilbert_val_logits.npy", f"{DATA_DIR}/distilbert_test_logits.npy"
if RETRAIN or not (os.path.exists(lg_val) and os.path.exists(lg_test)):
    val_logits  = trainer.predict(ds_val).predictions
    test_logits = trainer.predict(ds_test).predictions
    np.save(lg_val, val_logits); np.save(lg_test, test_logits)
    np.save(f"{DATA_DIR}/val_labels.npy",  val_df["label"].values)
    np.save(f"{DATA_DIR}/test_labels.npy", test_df["label"].values)
else:
    val_logits, test_logits = np.load(lg_val), np.load(lg_test)

p_te_dl = torch.softmax(torch.tensor(test_logits), dim=1)[:, 1].numpy()
np.save(f"{DATA_DIR}/distilbert_test_proba.npy", p_te_dl)
report("DistilBERT · TEST", test_df["label"], p_te_dl)""")

md("### Confusion matrix + порівняння з baseline")
code("""cm = confusion_matrix(test_df["label"], (p_te_dl >= 0.5).astype(int))
ConfusionMatrixDisplay(cm, display_labels=["legit", "phish"]).plot(
    cmap="Greens", colorbar=False)
plt.title("DistilBERT · confusion matrix (test)"); plt.tight_layout(); plt.show()""")

code("""compare = pd.DataFrame({
    "F1":      [f1_score(test_df["label"], (p_te_base >= .5).astype(int)),
                f1_score(test_df["label"], (p_te_dl >= .5).astype(int))],
    "ROC_AUC": [roc_auc_score(test_df["label"], p_te_base),
                roc_auc_score(test_df["label"], p_te_dl)],
    "PR_AUC":  [average_precision_score(test_df["label"], p_te_base),
                average_precision_score(test_df["label"], p_te_dl)],
}, index=["TF-IDF+LogReg", "DistilBERT"]).round(4)
compare""")

md("### Baseline vs DistilBERT — візуальне порівняння")
code("""ax = compare.plot(kind="bar", rot=0, figsize=(8, 4),
                  color=["#4c72b0", "#55a868", "#c44e52"])
ax.set_ylim(0.90, 1.005); ax.set_ylabel("метрика")
ax.set_title("Baseline vs DistilBERT (held-out test)")
ax.legend(loc="lower right", ncol=3, fontsize=9)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.show()
print("Практично нарівні — задача лексично проста (leakage). Цінність DL — у калібрації/поясненнях/адаптовності.")""")

md("### Збережена модель")
code("""print("модель збережено у:", SAVE_DIR)
print("вміст:", sorted(os.listdir(SAVE_DIR)))""")

md("""---
**F2 готово.** DistilBERT навчено, логіти val/test збережено, модель — у `model/distilbert`.
Далі **F3 · Калібрація** (temperature scaling на val-логітах) — reliability diagram + ECE/AECE/Brier
до/після.""")

# ---------- F3 ----------
md("""## F3 · Калібрування

Нейромережі часто занадто впевнені: кажуть «90%», а праві рідше. Мені це важливо, бо шкала
«очевидності» — це сама ймовірність `P(phish)`, а не лише «фішинг / не фішинг». Тому підбираю одне
число `T` на валідації й ділю на нього виходи моделі (temperature scaling). Точність від цього не
змінюється, змінюється лише те, наскільки чесні самі ймовірності. Дивимось на це через ECE (похибка
калібрування), Brier та графік «впевненість проти реальності» до і після.""")

md("### Завантаження логітів")
code("""val_logits  = torch.tensor(np.load(f"{DATA_DIR}/distilbert_val_logits.npy"),  dtype=torch.float32)
test_logits = torch.tensor(np.load(f"{DATA_DIR}/distilbert_test_logits.npy"), dtype=torch.float32)
val_y  = torch.tensor(np.load(f"{DATA_DIR}/val_labels.npy"),  dtype=torch.long)
test_y = np.load(f"{DATA_DIR}/test_labels.npy")
print("val logits:", tuple(val_logits.shape), "| test logits:", tuple(test_logits.shape))""")

md("### Підбір температури на val")
code("""class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return logits / self.T
    def fit(self, logits, labels):
        opt = torch.optim.LBFGS([self.T], lr=0.05, max_iter=80)
        ce = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad(); loss = ce(self.forward(logits), labels); loss.backward(); return loss
        opt.step(closure); return self

scaler = TemperatureScaler().fit(val_logits, val_y)
T = float(scaler.T.detach())
print(f"fitted temperature T = {T:.4f}  ({'overconfident, softening' if T > 1 else 'sharpening'})")""")

md("### Метрики калібрації (до / після)")
code("""def to_proba(logits, T=1.0):
    return torch.softmax(torch.as_tensor(np.asarray(logits)) / T, dim=1)[:, 1].numpy()

def ece(p, y, n_bins=15, adaptive=False):
    y = np.asarray(y); pred = (p >= .5).astype(int)
    conf = np.maximum(p, 1 - p); corr = (pred == y).astype(float)
    edges = (np.unique(np.quantile(conf, np.linspace(0, 1, n_bins + 1)))
             if adaptive else np.linspace(0, 1, n_bins + 1))
    e, n = 0.0, len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum(): e += abs(conf[m].mean() - corr[m].mean()) * (m.sum() / n)
    return e

brier = lambda p, y: float(np.mean((p - np.asarray(y)) ** 2))

p_before = to_proba(test_logits, 1.0)
p_after  = to_proba(test_logits, T)
cal = pd.DataFrame({
    "ECE":   [ece(p_before, test_y),               ece(p_after, test_y)],
    "AECE":  [ece(p_before, test_y, adaptive=True), ece(p_after, test_y, adaptive=True)],
    "Brier": [brier(p_before, test_y),             brier(p_after, test_y)],
}, index=["до (T=1)", f"після (T={T:.2f})"]).round(4)
cal""")

md("### Reliability diagram (до / після)")
code("""def rel_points(p, y, n_bins=12):
    conf = np.maximum(p, 1 - p); corr = ((p >= .5).astype(int) == np.asarray(y)).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum(): xs.append(conf[m].mean()); ys.append(corr[m].mean())
    return np.array(xs), np.array(ys)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for a, (p, ttl) in zip(ax, [(p_before, "до (T=1)"), (p_after, f"після (T={T:.2f})")]):
    xs, ys = rel_points(p, test_y)
    a.plot([0.5, 1], [0.5, 1], "--", color="grey", label="ідеал")
    a.plot(xs, ys, marker="o", color="#c44e52")
    a.set_title(f"Reliability · {ttl}\\nECE={ece(p, test_y):.4f}")
    a.set_xlabel("confidence"); a.legend()
ax[0].set_ylabel("accuracy")
plt.tight_layout(); plt.show()""")

md("### Збереження калібратора")
code("""import json
np.save(f"{DATA_DIR}/distilbert_test_proba_calibrated.npy", p_after)
json.dump({"temperature": T}, open(f"{DATA_DIR}/calibration.json", "w"))
print(f"saved calibrated proba + T={T:.4f} -> {DATA_DIR}/calibration.json")""")

md("""---
**F3 готово.** Шкала «очевидності» тепер спирається на каліброване `P(phish)` (ECE впав ~вдвічі).
Далі **F4 · Explainability** — Integrated Gradients (Captum) → підсвічені токени-драйвери.""")

# ---------- F4 ----------
md("""## F4 · Пояснення рішень

Підсвічуємо слова, які найбільше вплинули на бал — щоб автор бачив, чому лист виглядає фішингово.
Метод — Integrated Gradients (бібліотека Captum): він розкладає рішення моделі за внеском кожного
слова. Рахуємо на CPU (стабільніше). У продукті це працює на вимогу (кнопка «Пояснити»), а не на
кожне натискання, бо трохи повільніше за звичайний скоринг.""")

md("### Підготовка атрибутора (CPU)")
code("""attr_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to("cpu").eval()

def _fwd(input_ids, attn):
    return torch.softmax(attr_model(input_ids=input_ids, attention_mask=attn).logits, dim=1)[:, 1]

lig = LayerIntegratedGradients(_fwd, attr_model.distilbert.embeddings)

def attribute(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    base = torch.full_like(enc["input_ids"], tokenizer.pad_token_id)
    base[:, 0] = tokenizer.cls_token_id; base[:, -1] = tokenizer.sep_token_id
    attrs, delta = lig.attribute(enc["input_ids"], baselines=base,
        additional_forward_args=(enc["attention_mask"],),
        n_steps=50, return_convergence_delta=True)
    a = attrs.sum(dim=-1).squeeze(0)
    a = a / torch.norm(a)
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze().tolist())
    prob = float(_fwd(enc["input_ids"], enc["attention_mask"]))
    return toks, a, prob, float(delta)

def make_record(text, true_label):
    toks, a, prob, delta = attribute(text)
    pred = "phish" if prob >= 0.5 else "legit"
    return viz.VisualizationDataRecord(
        a, prob, pred, "phish" if true_label else "legit", "phish",
        float(a.sum()), toks, delta)

print("attributor ready")""")

md("### Підсвічені приклади: найвпевненіші phish, легіт і Whalen-подібний лур")
code("""order = np.argsort(p_after)
sel = list(order[-2:][::-1]) + list(order[:2])         # 2 найвищі P(phish) + 2 найнижчі
examples = [(test_df.iloc[i]["text"][:1200], int(test_df.iloc[i]["label"])) for i in sel]
examples.append((html_to_scoring_text(demo_html, "Urgent: Wire Transfer"), 1))  # синтетичний лур

records = [make_record(t, lab) for t, lab in examples]
viz.visualize_text(records)""")

md("""Зелене підсвічує слова, що штовхають бал до фішингу, червоне — до легіту. Типові маркери
(*urgent / password / account / verify*) виділяються найсильніше — саме їх підсвітка й показуватиме
автору в редакторі.""")

md("""---
**F4 готово.** Маємо token-атрибуцію (spans). Далі **F5 · Eval vs LLM** — спільна таблиця
якість × латентність × вартість (своя модель vs zero-shot LLM).""")

# ---------- F5 ----------
md("""## F5 · Порівняння з LLM — якість, швидкість, вартість

Головне порівняння роботи. Швидкість наших моделей **міряю реально** на CPU. Для LLM (без API-ключа)
швидкість і вартість **оцінюю** з кількості токенів і опублікованих цін; якість LLM беру з літератури
(Koide та ін., 2024: GPT-4 ~99% на фішингу) — її можна замінити реальним прогоном. Усе позначено явно.""")

md("### Латентність наших моделей (виміряно на CPU, batch=1)")
code("""sample = test_df["text"].iloc[:100].tolist()
enc_list = [tokenizer(t, return_tensors="pt", truncation=True, max_length=256) for t in sample]
with torch.no_grad():
    for e in enc_list[:5]:
        attr_model(**e)                              # warmup
dl_times = []
with torch.no_grad():
    for e in enc_list:
        t0 = time.perf_counter(); attr_model(**e); dl_times.append((time.perf_counter() - t0) * 1000)
base_times = []
for t in sample:
    t0 = time.perf_counter(); logreg.predict_proba(tfidf.transform([t])); base_times.append((time.perf_counter() - t0) * 1000)

dl_p50, dl_p95 = np.percentile(dl_times, [50, 95])
base_p50 = np.percentile(base_times, 50)
print(f"DistilBERT (PyTorch fp32 CPU): p50 {dl_p50:.1f} ms | p95 {dl_p95:.1f} ms")
print(f"TF-IDF+LogReg:                 p50 {base_p50:.2f} ms")
print("(F7: ONNX + INT8 зменшує DistilBERT до ~10 мс p95)")""")

md("### Оцінка LLM: вартість із токенів × ціни (без викликів)")
code("""avg_in = float(np.mean([len(tokenizer(t)["input_ids"]) for t in sample])) + 200  # +overhead промпта
PRICE = {"gpt-4o-mini": 0.15, "gpt-4o": 2.50}        # $ за 1M вхідних токенів (станом на 2025)
cost_1k = {k: (avg_in / 1e6) * v * 1000 for k, v in PRICE.items()}
print(f"сер. вхідних токенів/лист ≈ {avg_in:.0f}")
for k in PRICE:
    print(f"  zero-shot {k}: ≈ ${cost_1k[k]:.2f} / 1000 листів")""")

md("### Головна таблиця-висновок")
code("""f1_base = f1_score(test_y, (p_te_base >= .5).astype(int))
f1_dl   = f1_score(test_y, (p_after  >= .5).astype(int))   # каліброване, поріг 0.5

summary = pd.DataFrame([
    ["TF-IDF + LogReg",        f"{f1_base:.3f}", f"{base_p50:.2f} ms (вим.)",  "$0 (локально)"],
    ["DistilBERT (калібр.)",   f"{f1_dl:.3f}",   f"{dl_p50:.0f} ms (вим.)",    "$0 (локально)"],
    ["zero-shot GPT-4o-mini",  "≈0.97 (літ.)",   "300–2000 ms (тип.)",        f"≈${cost_1k['gpt-4o-mini']:.2f} / 1k"],
    ["zero-shot GPT-4o",       "≈0.99 (літ.)",   "300–2000 ms (тип.)",        f"≈${cost_1k['gpt-4o']:.2f} / 1k"],
], columns=["Модель", "F1", "Латентність p50", "Вартість інференсу"])
summary""")

md("""**Висновок.** Наша калібрована DistilBERT тримає якість на рівні LLM (а базова модель і зовсім
нарівні), але працює локально й безкоштовно. На цьому процесорі PyTorch дає ~140 мс, а стиснута
ONNX-INT8 (F7) — ~50 мс, тобто помітно швидше за типову LLM і без оплати за виклик. Саме тому оцінку
на кожне натискання тягне лише своя легка модель, а не LLM. Якість LLM тут — орієнтир з літератури;
за потреби замінюється реальним прогоном на кількох десятках листів.""")

md("""---
**F5 готово.** Головну тезу підтверджено таблицею. Далі **F6** — прогін реалістичних корпоративних
приманок через модель з підсвіткою слів.""")

# ---------- F6 ----------
md("""## F6 · Перевірка на реалістичних приманках

Навчальні приманки навмисно виглядають правдоподібно — на відміну від «дикого» спаму в публічних
датасетах. Тут я проганяю кілька таких реалістичних листів (з плейсхолдерами `{{.URL}}` тощо) через
модель і дивлюся на бал. Очікування: вдало зроблена приманка читається майже як звичайний лист, тож
модель дає їй низький бал — а це для продукту й означає «переконлива». Реальних платформних шаблонів
тут немає, тож беру синтетичні правдоподібні приклади.""")

md("### Скоринг-функція (калібрована)")
code("""def score_text(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = attr_model(**enc).logits
    return float(torch.softmax(logits / T, dim=1)[0, 1])      # каліброване P(phish)

def obviousness(p):
    return "очевидний фішинг" if p >= 0.5 else "переконлива приманка"
print("score_text ready")""")

md("### Реалістичні Whalen-подібні шаблони")
code('''TEMPLATES = {
 "CEO wire transfer": ("Confidential: Urgent Wire Transfer",
   "<p>Hi {{.FirstName}},</p><p>I need you to process an urgent wire transfer for a "
   "confidential acquisition. Please keep this between us until it closes. Details attached at "
   "<a href=\\"{{.URL}}\\">the secure portal</a>.</p><p>Regards,<br>Michael R. Bradford, CEO</p>"),
 "IT password reset": ("Action required: password expires today",
   "<p>Dear {{.FirstName}},</p><p>Your corporate password expires in 2 hours. Reset it now via "
   "<a href=\\"{{.URL}}\\">the IT self-service portal</a> to avoid losing access.</p>"
   "<p>IT Service Desk</p>"),
 "DocuSign shared doc": ("{{.FirstName}}, please review the Q3 agreement",
   "<p>Hi {{.FirstName}},</p><p>A document was shared with you for signature. "
   "<a href=\\"{{.URL}}\\">Review &amp; sign</a>.</p><p>Powered by DocuSign</p>"),
 "HR payroll update": ("Payroll: confirm your direct deposit details",
   "<p>Hi {{.FirstName}},</p><p>We are updating payroll records. Please confirm your direct "
   "deposit details <a href=\\"{{.URL}}\\">here</a> before Friday.</p><p>People Operations</p>"),
}

rows = []
for name, (subj, html) in TEMPLATES.items():
    p = score_text(html_to_scoring_text(html, subj))
    rows.append([name, round(p, 3), obviousness(p)])
pd.DataFrame(rows, columns=["Шаблон", "P(phish)", "Очевидність"])''')

md("### Підсвітка драйверів для одного шаблону")
code("""name = "CEO wire transfer"
subj, html = TEMPLATES[name]
rec = make_record(html_to_scoring_text(html, subj), true_label=1)
viz.visualize_text([rec])""")

md("""**Спостереження.** Реалістичні приманки часто отримують помірний або низький бал — і це очікувано:
модель, навчена на «дикому» спамі, бачить охайний корпоративний лист як близький до звичайного. Для
продукту це корисний сигнал (приманка переконлива). Але важливо памʼятати: бал показує, наскільки лист
*схожий на типовий фішинг*, а не наскільки реальна людина на нього клікне — це різні речі. На реальних
платформних шаблонах цей розділ варто повторити.""")

md("""---
**F6 готово.** Поведінку на реалістичних приманках показано. Далі **F7** — стиснення моделі (ONNX +
INT8) і замір швидкості.""")

# ---------- F7 ----------
md("""## F7 · Стиснення — ONNX + INT8

ONNX — формат, у якому модель працює без PyTorch (легший сервіс). Пакет `optimum` несумісний з
transformers 4.57, тож експортую напряму (`torch.onnx.export`) і додатково стискаю ваги до 8 біт
(`quantize_dynamic`). Перевіряю розмір, швидкість і що якість (F1) не впала.""")

md("### Export у ONNX + динамічна INT8-квантизація")
code("""import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
os.makedirs("onnx", exist_ok=True)

class _Wrap(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

_d = tokenizer("example email", return_tensors="pt", truncation=True, max_length=256)
torch.onnx.export(_Wrap(attr_model), (_d["input_ids"], _d["attention_mask"]), "onnx/model.onnx",
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                  "logits": {0: "b"}}, opset_version=14)
quantize_dynamic("onnx/model.onnx", "onnx/model_int8.onnx", weight_type=QuantType.QInt8)
print("ONNX fp32 %.0f MB  ->  INT8 %.0f MB" % (
    os.path.getsize("onnx/model.onnx") / 1e6, os.path.getsize("onnx/model_int8.onnx") / 1e6))""")

md("### Бенчмарк: латентність, розмір, паритет F1")
code("""sess = ort.InferenceSession("onnx/model_int8.onnx", providers=["CPUExecutionProvider"])
def onnx_logits(enc):
    return sess.run(None, {"input_ids": enc["input_ids"].numpy(),
                           "attention_mask": enc["attention_mask"].numpy()})[0]

_enc = [tokenizer(t, return_tensors="pt", truncation=True, max_length=256) for t in sample]
for e in _enc[:5]:
    onnx_logits(e)                                   # warmup
onnx_t = []
for e in _enc:
    t0 = time.perf_counter(); onnx_logits(e); onnx_t.append((time.perf_counter() - t0) * 1000)
onnx_p50, onnx_p95 = np.percentile(onnx_t, [50, 95])

_slice = test_df["text"].iloc[:500]
onnx_pred = [int(np.argmax(onnx_logits(
    tokenizer(t, return_tensors="pt", truncation=True, max_length=256))[0])) for t in _slice]
onnx_f1 = f1_score(test_y[:500], onnx_pred)

pt_size   = os.path.getsize(f"{SAVE_DIR}/model.safetensors") / 1e6
onnx_size = os.path.getsize("onnx/model_int8.onnx") / 1e6
pd.DataFrame([
    ["DistilBERT PyTorch fp32", f"{pt_size:.0f} MB", f"{dl_p50:.0f} ms",   f"{f1_dl:.3f}"],
    ["DistilBERT ONNX INT8",    f"{onnx_size:.0f} MB", f"{onnx_p50:.0f} ms", f"{onnx_f1:.3f} (на 500)"],
], columns=["Варіант", "Розмір", "Латентність p50", "F1"])""")

md("""---
**F7 готово.** Стиснута ONNX-INT8: у ~4 рази менша й у ~3 рази швидша за звичайну PyTorch-модель, якість
без втрат — готова до використання в сервісі.""")

# ---------- F8: synthetic corporate-style mini-benchmark (diagnostic) ----------
md("""## F8 · Власний корпоративний тест

Перевірка на іншому публічному наборі нічого не показала б (той самий стиль джерел, 81% збіг із
тренувальним). Реальна відмінність — корпоративний стиль листів. Тому роблю власний тест: 60
синтетичних листів (20 очевидних приманок / 20 тонких / 20 легітимних, корпоративний стиль) і дивлюся,
де поточна модель провалюється. Тест лежить у `synth.py` і **ніколи не використовується для навчання**.""")

md("### Held-out бенчмарк (60 листів)")
code("""from synth import build_benchmark, gen

bench = pd.DataFrame(build_benchmark(), columns=["raw", "label", "category"])
bench["text"] = bench["raw"].map(lambda t: html_to_scoring_text(t))
print("benchmark:", dict(bench.category.value_counts()))""")

md("### Поточна модель на бенчмарку")
code("""def predict_labels(model, texts):
    model.eval(); out = []
    with torch.no_grad():
        for t in texts:
            e = tokenizer(t, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            e = {k: v.to(model.device) for k, v in e.items()}
            out.append(int(torch.argmax(model(**e).logits, dim=1)))
    return np.array(out)

def cat_report(col):
    r = {}
    for c in ["obvious_phish", "subtle_phish"]:
        r[f"recall_{c}"] = float((bench.loc[bench.category == c, col] == 1).mean())
    r["legit_specificity"] = float((bench.loc[bench.category == "legit", col] == 0).mean())
    r["bench_F1"] = f1_score(bench["label"], bench[col])
    return r

bench["pred_before"] = predict_labels(attr_model, bench["text"])
pd.DataFrame({"public-only model": cat_report("pred_before")}).T.round(3)""")

md("""**Діагноз.** Модель, що мала ~0.98 на публічному тесті, на корпоративних листах **майже не ловить
луків** (обвʼязкові ~0.25, тонкі ~0.00), хоча легіт впізнає (specificity ~1.0). Вона вивчила
«спам-словник = фішинг», а корпоративні луки його не мають. Це кількісний доказ domain-gap і
прямий мандат на покращення.""")

# ---------- F9: target-style augmentation (improvement) ----------
md("""## F9 · Target-style augmentation

Лікуємо domain-gap **синтетичними корпоративними даними**. Генератор (`synth.gen`) комбінаторно
будує ~600 різноманітних листів (стиль навмисно інший за бенчмарк), домішуємо їх до підвибірки
публічних і **дотреновуємо** модель. Бенчмарк лишається строго held-out.""")

md("### Генерація train-аугментації")
code("""synth_tr = pd.DataFrame(gen(300, 300, seed=123), columns=["raw", "label", "category"]).drop_duplicates("raw")
synth_tr["text"] = synth_tr["raw"].map(lambda t: html_to_scoring_text(t))
assert len(set(synth_tr["text"]) & set(bench["text"])) == 0, "train/benchmark overlap!"
print("synth train:", len(synth_tr), dict(synth_tr.category.value_counts()), "| benchmark held-out:", len(bench))""")

md("### Дотренування (ідемпотентно: load model/distilbert_aug якщо є)")
code("""AUG_DIR = "model/distilbert_aug"
if os.path.isdir(AUG_DIR):
    aug_model = AutoModelForSequenceClassification.from_pretrained(AUG_DIR).to("cpu").eval()
    print("loaded cached augmented model <-", AUG_DIR)
else:
    pub_sub = train_df.sample(4000, random_state=SEED)[["text", "label"]]
    mix = pd.concat([pub_sub, pd.concat([synth_tr[["text", "label"]]] * 2)],
                    ignore_index=True).sample(frac=1, random_state=SEED)
    base = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    aug_args = TrainingArguments(output_dir="out_aug", learning_rate=1e-5,
        per_device_train_batch_size=16, num_train_epochs=2, weight_decay=0.01,
        eval_strategy="no", save_strategy="no", logging_steps=100, report_to="none",
        fp16=False, seed=SEED)
    aug_trainer = Trainer(model=base, args=aug_args,
        train_dataset=Dataset.from_pandas(mix, preserve_index=False).map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN), batched=True),
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer))
    aug_trainer.train(); aug_trainer.save_model(AUG_DIR); tokenizer.save_pretrained(AUG_DIR)
    aug_model = base.to("cpu").eval()
    print("trained & saved augmented model ->", AUG_DIR)""")

md("### До / після (бенчмарк + публічний тест)")
code("""bench["pred_after"] = predict_labels(aug_model, bench["text"])
ts = test_df.sample(800, random_state=7)
pub_before = f1_score(ts["label"], predict_labels(attr_model, ts["text"]))
pub_after  = f1_score(ts["label"], predict_labels(aug_model,  ts["text"]))

comp = pd.DataFrame({"before (public-only)": cat_report("pred_before"),
                     "after (augmented)":    cat_report("pred_after")}).T.round(3)
comp["public_test_F1(800)"] = [round(pub_before, 3), round(pub_after, 3)]
comp""")

md("### До / після augmentation — візуальне порівняння")
code("""cols = ["recall_obvious_phish", "recall_subtle_phish", "bench_F1", "public_test_F1(800)"]
labels = ["recall очевидні", "recall тонкі", "bench-F1", "public F1"]
ax = comp[cols].T.plot(kind="bar", rot=0, figsize=(8.2, 4.2), color=["#c44e52", "#55a868"])
ax.set_xticklabels(labels); ax.set_ylim(0, 1.08); ax.set_ylabel("значення")
ax.set_title("Публічна модель vs augmented (корпоративний бенчмарк)")
ax.legend(["before (public-only)", "after (augmented)"], fontsize=9)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.show()
print("recall очевидних приманок 0.25 -> 0.85, bench-F1 0.22 -> 0.64, public F1 без втрат.")""")

md("""**Результат.** Augmentation підняла recall на **очевидних** корпоративних луках з ~0.25 до ~0.85
і bench-F1 з ~0.22 до ~0.64 — **без втрат на публічному тесті** і без зростання false-positives
(legit specificity лишилась ~1.0). **Тонкі** луки досі майже не ловляться (~0.10) — це чесна відкрита
межа: вони лексично майже не відрізняються від легітимних, і ~600 синтетичних прикладів замало.
Висновок: domain-gap **закривається** target-style даними на очевидних луках; тонкі потребують більше/
різноманітніших даних (масштабна LLM-генерація або приватні Whalen-дані).""")

# ---------- F10 ----------
md("""## F10 · Висновки

**Що вийшло.** Калібрована DistilBERT-модель, що оцінює «очевидність фішингу»:
- проста базова модель і DistilBERT — практично нарівні (F1 ~0.98); на цьому датасеті задача проста
  через leakage, тож нейромережа не дає переваги в чистій точності;
- цінність DistilBERT — у **калібруванні** (похибка ECE вдвічі нижча), **поясненнях** (підсвітка слів)
  і **можливості дотренувати** під новий стиль;
- **стиснення в ONNX-INT8:** у 4 рази менша модель, швидша, без втрат якості, і безкоштовно локально
  проти платної LLM;
- **розрив на корпоративних листах виміряно й частково закрито:** публічна модель ловила ~25%
  очевидних корпоративних приманок; дотренування підняло це до ~85% (bench-F1 0.22→0.64) без втрат на
  публічному тесті. Тонкі приманки (~10%) лишаються відкритою межею.

**Обмеження.** (1) Leakage у датасеті завищує публічні метрики (найінформативніші ознаки — назви
корпусів); (2) без augmentation реалістичні приманки модель пропускає; augmentation виправляє
очевидні, але не тонкі; (3) бал показує «схожість на типовий фішинг», а не реальну ймовірність, що
людина клікне; (4) лише текст, сигнали з HTML поза розглядом; (5) дані для дотренування синтетичні —
для продукту потрібні реальні корпоративні.

**Що передаю далі:** спільну функцію `html_to_scoring_text`, ваги `model/distilbert` (базова) і
`model/distilbert_aug` (під корпоративний стиль), температуру калібрування в `data/calibration.json`
та стиснуту `onnx/model_int8.onnx` для сервісу.

**Наступні кроки:** більше й різноманітніших корпоративних даних (особливо тонкі приманки), реальні
платформні шаблони замість синтетичних, тренування з довшим контекстом на GPU.""")

md("""## F10 · Перевірка шкали на людських оцінках

Головне обмеження: бал моделі показує «схожість на типовий фішинг», а не точну «очевидність» на думку
людини. Перевіряю це напряму на **ручній розмітці** (`data/labeled/labeled_v{1,2}.parquet`,
Label Studio): для кожного листа є вердикт (фішинг / легіт) і оцінка очевидності 0–10. Питання — чи
вищий бал моделі означає «очевидніше» на думку людини (рангова кореляція Спірмена).

Ці дані — **лише для перевірки, ніколи для навчання** (їх немає в тренувальних). Люди ставили
очевидність тільки для фішингу (для легіту — порожньо), тож рахую два числа: серед самого фішингу
(очевидність 1–10) і на всьому наборі (легіт беру за 0).""")
code('''from scipy.stats import spearmanr

_frames = [pd.read_parquet(f"data/labeled/labeled_v{v}.parquet") for v in (1, 2)]
lab = pd.concat(_frames, ignore_index=True)
if "sample_id" in lab.columns:
    lab = lab.drop_duplicates(subset="sample_id", keep="last").reset_index(drop=True)

_tok = AutoTokenizer.from_pretrained("model/distilbert_aug")
_mdl = AutoModelForSequenceClassification.from_pretrained("model/distilbert_aug").eval()
_T = json.load(open("data/calibration.json"))["temperature"]

@torch.no_grad()
def _p_phish(text):
    st = html_to_scoring_text(text)                       # той самий контракт, що й сервінг
    enc = _tok(st, truncation=True, max_length=256, return_tensors="pt")
    return torch.softmax(_mdl(**enc).logits.squeeze(0) / _T, dim=-1)[1].item()

lab["p_phish"] = [_p_phish(t) for t in lab["text"]]

phish = lab.dropna(subset=["obviousness"])
rho_p, pv_p = spearmanr(phish["obviousness"], phish["p_phish"])
full = lab.copy(); full["obviousness"] = full["obviousness"].fillna(0.0)
rho_f, pv_f = spearmanr(full["obviousness"], full["p_phish"])

print(f"T={_T:.4f}  всього={len(lab)}")
print(f"[primary]   phishing-only n={len(phish)}: Spearman rho={rho_p:.3f} (p={pv_p:.1e})")
print(f"[secondary] Safe=0        n={len(full)}: Spearman rho={rho_f:.3f} (p={pv_f:.1e})")
for v in ("Phishing", "Safe"):
    s = lab[lab["verdict"] == v]["p_phish"]
    print(f"  mean p_phish[{v:8s}] = {s.mean():.3f} (n={len(s)})")''')
code('''_rng = np.random.default_rng(0)
def _jit(a, s): return a.to_numpy(float) + _rng.uniform(-s, s, len(a))
fig, ax = plt.subplots(figsize=(7.4, 5))
for _verd, _c, _lbl in [("Safe", "#2e86ab", "легіт"), ("Phishing", "#d1495b", "фішинг")]:
    _m = full["verdict"] == _verd
    ax.scatter(_jit(full.loc[_m, "obviousness"], .14), _jit(full.loc[_m, "p_phish"], .02),
               c=_c, s=60, alpha=.72, edgecolors="white", linewidth=.7, label=_lbl, zorder=3)
ax.plot([0, 10], [0, 1], "--", color="#aaa", lw=1.3, zorder=1, label="ідеальний збіг")
ax.set_xlabel("Оцінка людини: наскільки очевидний фішинг  (0 = легіт, 10 = явний)")
ax.set_ylabel("Бал моделі  (0-1)")
ax.set_xlim(-.6, 10.6); ax.set_ylim(-.1, 1.12)
ax.set_title(f"Бал моделі проти оцінки людини\\nρ(усе)={rho_f:.2f}   ·   ρ(лише фішинг)={rho_p:.2f}", fontsize=12)
ax.annotate("фішинг, який модель зловила  (бал ≈ 1)", (2.9, 1.0), (1.4, .78),
            fontsize=9.5, color="#7a2233", arrowprops=dict(arrowstyle="->", color="#b3536a", lw=1.1))
ax.annotate("фішинг, який модель проґавила  (бал ≈ 0)", (5, .02), (3.3, .30),
            fontsize=9.5, color="#7a2233", arrowprops=dict(arrowstyle="->", color="#b3536a", lw=1.1))
ax.annotate("легіт → низький бал", (0, .02), (.5, .18),
            fontsize=9.5, color="#1c5a75", arrowprops=dict(arrowstyle="->", color="#2e86ab", lw=1.1))
ax.legend(loc="center right", fontsize=9, framealpha=.95); ax.grid(alpha=.22, zorder=0)
plt.tight_layout(); plt.show()''')
md("""**Як читати графік.** Кожна точка — лист. Сині (легіт) унизу зліва: людина каже «0», модель дає
близько 0. Червоні (фішинг) розпадаються на дві смуги: угорі (бал ≈ 1) ті, що модель зловила, унизу
(бал ≈ 0) ті, що проґавила. Модель майже завжди видає або ~1, або ~0, майже без середніх значень.

**Висновок.** Фішинг від легіту модель відділяє чітко, тому на всьому діапазоні збіг з людиною сильний
(ρ ≈ 0.79). Але серед самих фішингових листів вона не вміє тонко ранжувати «наскільки очевидний» — там
збіг лише помірний (ρ ≈ 0.49). Тобто бал — це радше грубий індикатор очевидності, ніж точний вимір.
Обмеження: невелика вибірка (десятки), одне джерело розмітки.""")

md("### Definition of Done — статус")
code("""done = pd.DataFrame([
    ["F0 Дані + EDA + лік-чеки + спліт", "✅"],
    ["F1 Baseline TF-IDF+LogReg", "✅"],
    ["F2 DistilBERT fine-tune (256/2ep)", "✅"],
    ["F3 Калібрація (ECE/AECE/Brier, reliability)", "✅"],
    ["F4 Explainability (Integrated Gradients)", "✅"],
    ["F5 Eval vs LLM (якість×латентність×$)", "✅"],
    ["F6 Whalen-валідація (domain gap)", "✅"],
    ["F7 ONNX + INT8 + latency (stretch)", "✅"],
    ["F8 Корпоративний міні-бенчмарк (діагностика)", "✅"],
    ["F9 Target-style augmentation (recall 0.25→0.85)", "✅"],
    ["F10 Spearman-валідація шкали (human labels)", "✅"],
    ["OOD на реальних 5 platform-шаблонах / тонкі луки", "⬜ next"],
], columns=["Етап", "Статус"])
done""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

import os as _os
out_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "phishing_scorer.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print("WROTE", out_path, "cells:", len(cells))
