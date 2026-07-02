"""
Movie Genre Classifier
======================
Predicts movie genre from plot descriptions using TF-IDF + multiple classifiers.
Best model is selected by cross-validation and evaluated on the test set.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time, re
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, f1_score
)
from sklearn.model_selection import cross_val_score
import joblib

# ─── 1. Load Data ───────────────────────────────────────────────────────────
print("=" * 65)
print("  MOVIE GENRE CLASSIFIER — TF-IDF + ML")
print("=" * 65)

print("\n📂 Loading data...")
train = pd.read_csv(
    'train_data.txt', sep=' ::: ', header=None,
    names=['id', 'title', 'genre', 'description'],
    engine='python', on_bad_lines='skip'
).dropna(subset=['description', 'genre'])

test = pd.read_csv(
    'test_data.txt', sep=' ::: ', header=None,
    names=['id', 'title', 'description'],
    engine='python', on_bad_lines='skip'
).dropna(subset=['description'])

solution = pd.read_csv(
    'test_data_solution.txt', sep=' ::: ', header=None,
    names=['id', 'title', 'genre', 'description'],
    engine='python', on_bad_lines='skip'
).dropna(subset=['genre'])

print(f"  Train samples : {len(train):,}")
print(f"  Test  samples : {len(test):,}")
print(f"  Genres        : {train['genre'].nunique()} unique labels")

# ─── 2. Text Preprocessing ──────────────────────────────────────────────────
def clean_text(text):
    """Lowercase, strip punctuation/numbers, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\n🔧 Preprocessing text...")
train['clean_desc'] = train['description'].apply(clean_text)
# Combine title + description for richer signal
train['features'] = train['title'].apply(clean_text) + ' ' + train['clean_desc']

test['clean_desc'] = test['description'].apply(clean_text)
test['features']   = test['title'].apply(clean_text) + ' ' + test['clean_desc']
solution['features'] = solution['title'].apply(clean_text) + ' ' + \
                        solution['description'].apply(clean_text)

X_train = train['features']
y_train = train['genre']
X_test  = solution['features']          # use solution file for evaluation
y_test  = solution['genre']

print(f"  Title + description features used for all models.")

# ─── 3. Define Pipelines ────────────────────────────────────────────────────
# Shared TF-IDF config
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    max_features=100_000,
    sublinear_tf=True,
    min_df=2,
    strip_accents='unicode',
    analyzer='word'
)

models = {
    "Logistic Regression": Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   LogisticRegression(
            C=5, max_iter=1000, solver='lbfgs', n_jobs=-1
        ))
    ]),
    "Linear SVM": Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   LinearSVC(C=1.0, max_iter=2000))
    ]),
    "Complement Naive Bayes": Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   ComplementNB(alpha=0.1))
    ]),
    "Multinomial Naive Bayes": Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   MultinomialNB(alpha=0.1))
    ]),
}

# ─── 4. Cross-Validation ────────────────────────────────────────────────────
print("\n📊 Running 5-fold cross-validation on training data...\n")
cv_results = {}
for name, pipe in models.items():
    t0 = time.time()
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=5, scoring='accuracy', n_jobs=-1)
    elapsed = time.time() - t0
    cv_results[name] = scores
    print(f"  {name:<30}  acc = {scores.mean():.4f} ± {scores.std():.4f}  ({elapsed:.1f}s)")

best_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\n  ✅ Best model: {best_name}")

# ─── 5. Train Best Model & Evaluate on Test Set ─────────────────────────────
print(f"\n🏋️  Training {best_name} on full training set...")
best_pipe = models[best_name]
best_pipe.fit(X_train, y_train)

print("🧪 Evaluating on test set...")
y_pred = best_pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f"\n  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Macro F1      : {macro_f1:.4f}")

# ─── 6. Also train all models for comparison ────────────────────────────────
print("\n📋 Test set accuracy — all models:")
test_accs = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    tacc = accuracy_score(y_test, preds)
    test_accs[name] = tacc
    print(f"  {name:<30}  {tacc:.4f}")

# ─── 7. Classification Report ───────────────────────────────────────────────
print(f"\n📄 Per-genre report — {best_name}:")
print(classification_report(y_test, y_pred, zero_division=0))

# ─── 8. Save Best Model ─────────────────────────────────────────────────────
joblib.dump(best_pipe, 'best_genre_model.pkl')
print("💾 Model saved to best_genre_model.pkl")

# ─── 9. Plots ───────────────────────────────────────────────────────────────
print("\n📈 Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Movie Genre Classifier — Model Analysis", fontsize=16, fontweight='bold')

# --- Plot 1: CV scores comparison ---
ax = axes[0, 0]
names = list(cv_results.keys())
means = [cv_results[n].mean() for n in names]
stds  = [cv_results[n].std()  for n in names]
short_names = [n.replace(" Naive Bayes", "\nNaive Bayes") for n in names]
bars = ax.barh(short_names, means, xerr=stds, capsize=5,
               color=['#4C72B0','#DD8452','#55A868','#C44E52'], alpha=0.85)
ax.set_xlabel("5-Fold CV Accuracy")
ax.set_title("Cross-Validation Accuracy by Model")
ax.set_xlim(0.5, 1.0)
for bar, val in zip(bars, means):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)

# --- Plot 2: Genre distribution in train ---
ax = axes[0, 1]
genre_counts = train['genre'].value_counts()
colors = plt.cm.tab20(np.linspace(0, 1, len(genre_counts)))
bars2 = ax.barh(genre_counts.index, genre_counts.values, color=colors, alpha=0.85)
ax.set_xlabel("Number of samples")
ax.set_title("Training Set — Genre Distribution")
ax.invert_yaxis()
for bar, val in zip(bars2, genre_counts.values):
    ax.text(val + 30, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va='center', fontsize=7)
ax.grid(axis='x', alpha=0.3)

# --- Plot 3: Confusion matrix (top 10 genres) ---
ax = axes[1, 0]
top_genres = train['genre'].value_counts().head(10).index.tolist()
mask_test = y_test.isin(top_genres)
cm = confusion_matrix(y_test[mask_test], y_pred[mask_test],
                      labels=top_genres, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=top_genres, yticklabels=top_genres,
            ax=ax, cbar=True, linewidths=0.5, annot_kws={"size": 7})
ax.set_title(f"Confusion Matrix — Top 10 Genres\n({best_name})")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)

# --- Plot 4: Per-genre F1 ---
ax = axes[1, 1]
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
genre_f1 = {k: v['f1-score'] for k, v in report.items()
            if k not in ('accuracy', 'macro avg', 'weighted avg')}
gf1_sorted = dict(sorted(genre_f1.items(), key=lambda x: x[1], reverse=True))
colors3 = ['#2ecc71' if v >= 0.6 else '#e67e22' if v >= 0.4 else '#e74c3c'
           for v in gf1_sorted.values()]
ax.barh(list(gf1_sorted.keys()), list(gf1_sorted.values()),
        color=colors3, alpha=0.85)
ax.axvline(0.6, color='green',  linestyle='--', alpha=0.5, label='0.60')
ax.axvline(0.4, color='orange', linestyle='--', alpha=0.5, label='0.40')
ax.set_xlabel("F1 Score")
ax.set_title(f"Per-Genre F1 Score — {best_name}")
ax.invert_yaxis()
ax.set_xlim(0, 1.0)
ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.3)
for i, (genre, f1) in enumerate(gf1_sorted.items()):
    ax.text(f1 + 0.01, i, f"{f1:.3f}", va='center', fontsize=7)

plt.tight_layout()
plt.savefig('genre_classifier_analysis.png', dpi=150, bbox_inches='tight')
print("  Saved genre_classifier_analysis.png")

# ─── 10. Top TF-IDF features per genre ─────────────────────────────────────
print("\n🔍 Top discriminative words per genre (Logistic Regression):")
lr_pipe = models["Logistic Regression"]
tfidf  = lr_pipe.named_steps['tfidf']
clf    = lr_pipe.named_steps['clf']
feat_names = np.array(tfidf.get_feature_names_out())
for i, genre in enumerate(clf.classes_):
    top_idx  = np.argsort(clf.coef_[i])[-8:][::-1]
    top_words = feat_names[top_idx]
    print(f"  {genre:<15}: {', '.join(top_words)}")

# ─── 11. Prediction function demo ───────────────────────────────────────────
print("\n🎬 Demo predictions on new plot summaries:")
demos = [
    ("Space Odyssey 2099",
     "A crew of astronauts embarks on a dangerous mission through a wormhole "
     "to find a new habitable planet for humanity as Earth faces ecological collapse."),
    ("Laugh Out Loud",
     "A bumbling accountant accidentally becomes a stand-up comedian after a "
     "series of hilarious misunderstandings at his company's annual talent show."),
    ("Shadow of the Killer",
     "A relentless detective hunts a masked serial killer who leaves cryptic "
     "clues at each murder scene, leading to a terrifying final confrontation."),
    ("Battlefield 1918",
     "Following a battalion of soldiers through the trenches of World War I, "
     "depicting the brutal reality of trench warfare and the bonds formed under fire."),
]

model = joblib.load('best_genre_model.pkl')
print(f"  (using saved {best_name} model)\n")
for title, plot in demos:
    inp = clean_text(title) + ' ' + clean_text(plot)
    pred = model.predict([inp])[0]
    print(f"  Title : {title}")
    print(f"  Pred  : {pred.upper()}")
    print()

print("=" * 65)
print(f"  FINAL RESULTS  ({best_name})")
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"  Macro F1      : {macro_f1:.4f}")
print("=" * 65)
