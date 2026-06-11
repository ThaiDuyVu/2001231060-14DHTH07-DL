"""
evaluate_model.py
─────────────────────────────────────────────────────────────
Chấm điểm model LSTM từ file test_data.json.
Hỗ trợ mở rộng dữ liệu: chỉ cần thêm entry vào JSON, script
tự động đọc hết và tính lại toàn bộ metrics + vẽ 4 biểu đồ.

Yêu cầu file trong cùng thư mục:
  best_model.pt · word2idx.json · label_map.json · test_data.json

Cài đặt:
  pip install torch matplotlib seaborn scikit-learn
"""

import json
import re
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ═══════════════════════════════════════════════════════════════
# 0. PATHS  —  chỉnh nếu file ở thư mục khác
# ═══════════════════════════════════════════════════════════════
BASE_DIR       = Path(__file__).parent
MODEL_PATH     = BASE_DIR / "best_model.pt"
WORD2IDX_PATH  = BASE_DIR / "word2idx.json"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"
TEST_DATA_PATH = BASE_DIR / "test_data.json"

# ═══════════════════════════════════════════════════════════════
# 1. HYPERPARAMETERS  —  khớp với lúc training
# ═══════════════════════════════════════════════════════════════
VOCAB_SIZE  = 10000
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
NUM_CLASSES = 4
MAX_LEN     = 50
DROPOUT     = 0.3

# ═══════════════════════════════════════════════════════════════
# 2. MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (hidden, _) = self.lstm(emb)
        out = self.dropout(hidden[-1])
        return self.fc(out)

# ═══════════════════════════════════════════════════════════════
# 3. LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════
def load_artifacts():
    for p in [MODEL_PATH, WORD2IDX_PATH, LABEL_MAP_PATH, TEST_DATA_PATH]:
        if not p.exists():
            sys.exit(f"[ERROR] File not found: {p}")

    with open(WORD2IDX_PATH, "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    label_map = {int(k): v for k, v in raw.items()}

    model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM,
                           NUM_LAYERS, NUM_CLASSES, DROPOUT)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    return model, word2idx, label_map, test_data

# ═══════════════════════════════════════════════════════════════
# 4. PREPROCESSING
# ═══════════════════════════════════════════════════════════════
STOPWORDS = {
    "the","to","a","and","i","of","is","you","that","in",
    "it","for","on","are","was","with","he","she","they",
    "we","be","this","have","do","at","by","not","but",
    "or","an","as","from","his","her","my","your","our",
    "so","if","up","out","about","what","which","who",
    "can","will","just","more","also","been","has","had",
    "its","their","there","then","than","when","no","one",
    "would","could","should","did","get","got","me","him",
    "them","all","were","said","how","go","like","im",
    "dont","its","very","much","some","into","after","over",
}

def tokenize(text: str) -> list:
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text.lower())
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 2]

def encode(text: str, word2idx: dict, max_len: int = MAX_LEN) -> list:
    tokens = tokenize(text)[:max_len]
    ids    = [word2idx.get(t, 1) for t in tokens]
    ids   += [0] * (max_len - len(ids))
    return ids

# ═══════════════════════════════════════════════════════════════
# 5. INFERENCE  —  per-comment probability averaging
# ═══════════════════════════════════════════════════════════════
def predict_one(text: str, model, word2idx) -> tuple:
    """Trả về (predicted_label_idx, confidence_score, full_probs)."""
    ids    = encode(text, word2idx)
    tensor = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].tolist()
    pred_idx   = int(np.argmax(probs))
    confidence = probs[pred_idx]
    return pred_idx, confidence, probs

# ═══════════════════════════════════════════════════════════════
# 6. RUN EVALUATION
# ═══════════════════════════════════════════════════════════════
def run_evaluation(model, word2idx, label_map, test_data):
    # Tạo reverse map: label_name → index
    name_to_idx = {v: k for k, v in label_map.items()}
    class_names = [label_map[i] for i in sorted(label_map.keys())]

    y_true, y_pred, confidences, all_probs = [], [], [], []

    print(f"\n{'─'*60}")
    print(f"  Evaluating {len(test_data)} samples...")
    print(f"{'─'*60}")

    for item in test_data:
        true_label  = item["label"]
        true_idx    = name_to_idx[true_label]
        pred_idx, conf, probs = predict_one(item["text"], model, word2idx)

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        confidences.append(conf)
        all_probs.append(probs)

    # ── Metrics ──────────────────────────────────────────────
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # ── Print summary ─────────────────────────────────────────
    print(f"\n  Overall Accuracy : {accuracy*100:.2f}%")
    print(f"  Total Samples    : {len(test_data)}")
    print(f"  Correct          : {int(accuracy*len(test_data))}")
    print(f"  Wrong            : {len(test_data) - int(accuracy*len(test_data))}")
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'─'*52}")
    for i, name in enumerate(class_names):
        print(f"  {name:<12} {precision[i]:>10.3f} {recall[i]:>10.3f} "
              f"{f1[i]:>10.3f} {int(support[i]):>10}")
    print(f"{'─'*60}\n")

    return {
        "accuracy"    : accuracy,
        "precision"   : precision,
        "recall"      : recall,
        "f1"          : f1,
        "support"     : support,
        "cm"          : cm,
        "confidences" : confidences,
        "y_true"      : y_true,
        "y_pred"      : y_pred,
        "class_names" : class_names,
        "n_samples"   : len(test_data),
    }

# ═══════════════════════════════════════════════════════════════
# 7. VISUALISATION  —  4 charts
# ═══════════════════════════════════════════════════════════════
# Palette nhất quán với SocialAI dark theme
COLORS = {
    "Tech"   : "#5ba8d8",
    "Sports" : "#e07b5a",
    "Finance": "#c8a84b",
    "Gaming" : "#9e6bdb",
}
BG      = "#0f0f0f"
BG2     = "#161616"
BORDER  = "#2a2a2a"
FG      = "#e8e8e8"
FG_DIM  = "#666666"

def apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor"  : BG,
        "axes.facecolor"    : BG2,
        "axes.edgecolor"    : BORDER,
        "axes.labelcolor"   : FG_DIM,
        "axes.titlecolor"   : FG,
        "xtick.color"       : FG_DIM,
        "ytick.color"       : FG_DIM,
        "text.color"        : FG,
        "grid.color"        : BORDER,
        "grid.linewidth"    : 0.5,
        "font.family"       : "monospace",
        "font.size"         : 10,
        "axes.titlesize"    : 11,
        "axes.titlepad"     : 12,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
    })

# ── Chart 1: Confusion Matrix ──────────────────────────────────
def plot_confusion_matrix(ax, cm, class_names):
    # Normalise theo row (true label)
    cm_norm = cm.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_norm = cm_norm / row_sum

    palette = sns.color_palette(
        ["#0d1e30", "#1a3a5f", "#2a6aaf", "#5ba8d8"], as_cmap=False
    )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "socialai_blue", ["#0d0d0d", "#1a3a5f", "#5ba8d8"]
    )

    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Label", labelpad=8)
    ax.set_ylabel("True Label",      labelpad=8)
    ax.set_title("1. Confusion Matrix  (row-normalised)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            raw = cm[i, j]
            pct = cm_norm[i, j]
            color = FG if pct < 0.5 else BG
            ax.text(j, i, f"{pct:.0%}\n({raw})",
                    ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=FG_DIM)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG_DIM, fontsize=8)
    cbar.set_label("Recall rate", color=FG_DIM, fontsize=8)

# ── Chart 2: F1 Score per class ────────────────────────────────
def plot_f1_bars(ax, f1, class_names):
    x      = np.arange(len(class_names))
    colors = [COLORS[c] for c in class_names]
    bars   = ax.bar(x, f1, color=colors, width=0.55,
                    linewidth=0, zorder=3)

    # Đường tham chiếu
    ax.axhline(y=0.7, color=BORDER, linewidth=1, linestyle="--", zorder=2)
    ax.text(len(class_names) - 0.05, 0.71, "0.70 baseline",
            ha="right", va="bottom", fontsize=7.5, color=FG_DIM)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 Score")
    ax.set_title("② F1 Score per Class")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, color=FG)

# ── Chart 3: Confidence Distribution ──────────────────────────
def plot_confidence_hist(ax, confidences, accuracy):
    n       = len(confidences)
    bins    = np.linspace(0, 1, 21)
    counts, edges = np.histogram(confidences, bins=bins)

    # Màu gradient theo độ cao
    max_c = max(counts) if max(counts) > 0 else 1
    for i in range(len(counts)):
        intensity = counts[i] / max_c
        color = matplotlib.colors.to_hex(
            (intensity * 0.36, intensity * 0.62, intensity * 0.85)
        )
        ax.bar(edges[i], counts[i],
               width=(edges[i+1]-edges[i]) * 0.9,
               align="edge", color=color, linewidth=0, zorder=3)

    # Đường accuracy
    ax.axvline(x=accuracy, color="#c8a84b", linewidth=1.5,
               linestyle="--", zorder=4)
    ax.text(accuracy + 0.01, ax.get_ylim()[1] * 0.92,
            f"acc={accuracy:.0%}", color="#c8a84b", fontsize=8)

    med = float(np.median(confidences))
    ax.axvline(x=med, color="#9e6bdb", linewidth=1.2,
               linestyle=":", zorder=4)
    ax.text(med + 0.01, ax.get_ylim()[1] * 0.78,
            f"median={med:.2f}", color="#9e6bdb", fontsize=8)

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Sample Count")
    ax.set_title("③ Confidence Distribution")
    ax.set_xlim(0, 1)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

# ── Chart 4: Overall Accuracy Gauge ────────────────────────────
def plot_accuracy_gauge(ax, accuracy, n_samples, class_names,
                        precision, recall, f1):
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.1, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("④ Overall Accuracy", pad=14)

    # ── Arc background (grey) ─────────────────────────────────
    theta_bg = np.linspace(np.pi, 0, 300)
    r = 1.0
    ax.plot(r * np.cos(theta_bg), r * np.sin(theta_bg),
            color=BORDER, linewidth=14, solid_capstyle="round", zorder=1)

    # ── Arc filled (colour based on accuracy) ─────────────────
    if accuracy >= 0.85:
        arc_color = "#4a9e4a"
    elif accuracy >= 0.65:
        arc_color = "#c8a84b"
    else:
        arc_color = "#c0392b"

    theta_fill = np.linspace(np.pi, np.pi - accuracy * np.pi, 300)
    ax.plot(r * np.cos(theta_fill), r * np.sin(theta_fill),
            color=arc_color, linewidth=14, solid_capstyle="round", zorder=2)

    # ── Big number ────────────────────────────────────────────
    ax.text(0, 0.30, f"{accuracy*100:.1f}%",
            ha="center", va="center",
            fontsize=34, fontweight="bold", color=FG)
    ax.text(0, 0.08, f"{int(accuracy*n_samples)} / {n_samples} correct",
            ha="center", va="center", fontsize=9, color=FG_DIM)

    # ── Macro average stats below ─────────────────────────────
    macro_p  = float(np.mean(precision))
    macro_r  = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    stats = [
        ("Macro Precision", macro_p),
        ("Macro Recall",    macro_r),
        ("Macro F1",        macro_f1),
    ]
    for idx, (lbl, val) in enumerate(stats):
        y_pos = -0.05 - idx * 0.17
        ax.text(-0.85, y_pos, lbl,
                ha="left", va="center", fontsize=8, color=FG_DIM)
        ax.text(0.85, y_pos, f"{val:.3f}",
                ha="right", va="center", fontsize=8,
                color=arc_color, fontweight="bold")

# ═══════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("\n  SocialAI · Model Evaluation")
    print("  " + "═" * 36)

    model, word2idx, label_map, test_data = load_artifacts()
    results = run_evaluation(model, word2idx, label_map, test_data)

    apply_dark_theme()

    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    fig.suptitle(
        f"SocialAI  ·  LSTM Model Evaluation  ·  {results['n_samples']} samples",
        fontsize=13, color=FG, y=0.98, fontfamily="monospace"
    )

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        left=0.07, right=0.96,
        top=0.93,  bottom=0.07,
        hspace=0.38, wspace=0.32,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_confusion_matrix(ax1, results["cm"],          results["class_names"])
    plot_f1_bars         (ax2, results["f1"],          results["class_names"])
    plot_confidence_hist (ax3, results["confidences"], results["accuracy"])
    plot_accuracy_gauge  (ax4, results["accuracy"],    results["n_samples"],
                          results["class_names"],      results["precision"],
                          results["recall"],           results["f1"])

    # Watermark
    fig.text(0.98, 0.01,
             f"test_data.json  ·  {results['n_samples']} samples",
             ha="right", va="bottom", fontsize=7.5, color="#333")

    plt.savefig("evaluation_report.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("  Chart saved → evaluation_report.png")

    plt.show()

if __name__ == "__main__":
    main()