"""
full_pipeline.py — Chạy toàn bộ từ Step 1 đến lưu file
Dùng trên Kaggle notebook: chạy 1 cell duy nhất
"""

import sqlite3
import pandas as pd
import re
import json
import os
import base64
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from IPython.display import HTML, display

# ============================================================
# BƯỚC 0: CẤU HÌNH — chỉnh ở đây nếu cần
# ============================================================
DB_PATH    = "/kaggle/input/datasets/organizations/kaggle/reddit-comments-may-2015/database.sqlite"
LIMIT      = 20000       # số comment mỗi nhãn
VOCAB_SIZE = 10000
MAX_LEN    = 50
EMBED_DIM  = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES= 4
DROPOUT    = 0.3
BATCH_SIZE = 64
NUM_EPOCHS = 7
LR         = 0.003
SEED       = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*55}")
print(f"  FULL PIPELINE — Device: {device}")
print(f"{'='*55}\n")

# ============================================================
# BƯỚC 1: LOAD DATA
# ============================================================
print("▶ BƯỚC 1: Load data...")
conn = sqlite3.connect(DB_PATH)

subreddit_groups = {
    "Tech"   : ["technology"],
    "Sports" : ["sports", "soccer", "nba"],
    "Finance": ["investing", "cryptocurrency"],
    "Gaming" : ["gaming"]
}

dfs = []
for label, subs in subreddit_groups.items():
    subs_str = ",".join([f"'{s}'" for s in subs])
    query = f"""
        SELECT author, subreddit, body, created_utc
        FROM May2015
        WHERE subreddit IN ({subs_str})
        ORDER BY RANDOM()
        LIMIT {LIMIT}
    """
    temp = pd.read_sql(query, conn)
    temp["label"] = label
    dfs.append(temp)

conn.close()
df = pd.concat(dfs, ignore_index=True)
print(f"   ✅ Loaded: {df.shape}")

# ============================================================
# BƯỚC 2+3: CLEAN + REMOVE NOISE
# ============================================================
print("\n▶ BƯỚC 2+3: Clean text...")

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_body"] = df["body"].apply(clean_text)
df = df[df["clean_body"].apply(lambda x: len(x.split()) >= 5)]
df = df.reset_index(drop=True)
print(f"   ✅ After clean: {df.shape}")
print(f"   {df['label'].value_counts().to_dict()}")

# ============================================================
# BƯỚC 6: USER AGGREGATION
# ============================================================
print("\n▶ BƯỚC 6: User aggregation...")

df_user_label = df.groupby(["author","label"]).size().reset_index(name="count")
df_majority   = df_user_label.loc[
    df_user_label.groupby("author")["count"].idxmax()
][["author","label"]]
df_majority.columns = ["author","interest"]

df_clean = df.merge(df_majority, on="author")
df_clean = df_clean[df_clean["label"] == df_clean["interest"]]
df_user  = (
    df_clean.groupby(["author","interest"])["clean_body"]
    .apply(" ".join)
    .reset_index()
)
df_user.columns = ["author","interest","aggregated_text"]
df_user = df_user[df_user["aggregated_text"].str.split().str.len() > 20]
df_user = df_user.reset_index(drop=True)
print(f"   ✅ User dataset: {df_user.shape}")
print(f"   {df_user['interest'].value_counts().to_dict()}")

# ============================================================
# BƯỚC 7: ENCODE LABEL + SPLIT
# ============================================================
print("\n▶ BƯỚC 7: Encode label + split...")

# Gán label_id theo thứ tự cố định — KHÔNG phụ thuộc vào data
LABEL_MAP     = {"Tech": 0, "Sports": 1, "Finance": 2, "Gaming": 3}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

df_user["label_id"] = df_user["interest"].map(LABEL_MAP)

train_df, temp_df = train_test_split(
    df_user, test_size=0.30,
    stratify=df_user["label_id"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50,
    stratify=temp_df["label_id"], random_state=SEED
)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"   ✅ Train:{len(train_df)} | Val:{len(val_df)} | Test:{len(test_df)}")
print(f"   Label map: {LABEL_MAP}")

# ============================================================
# BƯỚC 8: TOKENIZE + VOCAB
# ============================================================
print("\n▶ BƯỚC 8: Build vocabulary...")

STOPWORDS = set([
    "the","to","a","and","i","of","is","you","that","in",
    "it","for","on","are","was","with","he","she","they",
    "we","be","this","have","do","at","by","not","but",
    "or","an","as","from","his","her","my","your","our",
    "so","if","up","out","about","what","which","who",
    "can","will","just","more","also","been","has","had",
    "its","their","there","then","than","when","no","one",
    "would","could","should","did","get","got","me","him",
    "them","all","were","said","how","go","like","im",
    "dont","its","very","much","some","into","after","over"
])

def tokenize(text):
    tokens = text.lower().split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

counter = Counter()
for text in train_df["aggregated_text"]:
    counter.update(tokenize(text))

vocab    = ["<PAD>","<UNK>"] + [w for w,_ in counter.most_common(VOCAB_SIZE-2)]
word2idx = {w: i for i, w in enumerate(vocab)}

def encode(text, max_len=MAX_LEN):
    tokens = tokenize(text)[:max_len]
    ids    = [word2idx.get(t, 1) for t in tokens]
    ids   += [0] * (max_len - len(ids))
    return ids

print(f"   ✅ Vocab size: {len(word2idx)}")
print(f"   Top 5 từ: {[w for w,_ in counter.most_common(5)]}")

# ============================================================
# BƯỚC 9: DATASET + DATALOADER
# ============================================================
print("\n▶ BƯỚC 9: DataLoader...")

class RedditDataset(Dataset):
    def __init__(self, dataframe):
        self.texts  = [torch.tensor(encode(t), dtype=torch.long)
                       for t in dataframe["aggregated_text"]]
        self.labels = torch.tensor(dataframe["label_id"].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_loader = DataLoader(RedditDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(RedditDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(RedditDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False)

print(f"   ✅ Train:{len(train_loader)} | Val:{len(val_loader)} | Test:{len(test_loader)} batches")

# ============================================================
# BƯỚC 10: MODEL
# ============================================================
print("\n▶ BƯỚC 10: Build model...")

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded        = self.embedding(x)
        _, (hidden, _)  = self.lstm(embedded)
        out             = self.dropout(hidden[-1])
        return self.fc(out)

model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
total = sum(p.numel() for p in model.parameters())
print(f"   ✅ Model ready — {total:,} parameters")

# ============================================================
# BƯỚC 11: TRAINING
# ============================================================
print("\n▶ BƯỚC 11: Training...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss/len(loader), correct/total*100

best_val_acc = 0
history      = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

print(f"\n   {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
print(f"   {'-'*55}")

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total * 100
    val_loss, val_acc = evaluate(model, val_loader)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "/kaggle/working/best_model.pt")
        flag = "⭐"
    else:
        flag = ""

    print(f"   {epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_loss:>8.4f} | {val_acc:>6.2f}% {flag}")

print(f"\n   ✅ Training xong! Best Val Accuracy: {best_val_acc:.2f}%")

# Test set
_, test_acc = evaluate(model, test_loader)
print(f"   ✅ Test Accuracy: {test_acc:.2f}%")

# ============================================================
# BƯỚC 12: LƯU TẤT CẢ FILE + DOWNLOAD
# ============================================================
print("\n▶ BƯỚC 12: Lưu file...")

# Lưu word2idx
with open("/kaggle/working/word2idx.json", "w", encoding="utf-8") as f:
    json.dump(word2idx, f, ensure_ascii=False)

# Lưu label_map
with open("/kaggle/working/label_map.json", "w", encoding="utf-8") as f:
    json.dump(LABEL_MAP_INV, f, ensure_ascii=False)

# Xác nhận
for fname in ["best_model.pt", "word2idx.json", "label_map.json"]:
    size = os.path.getsize(f"/kaggle/working/{fname}")
    print(f"   ✅ {fname} — {size:,} bytes")

# Tạo link download
print("\n📥 Link tải file:")
files_to_download = [
    ("word2idx.json",  "application/json"),
    ("label_map.json", "application/json"),
    ("best_model.pt",  "application/octet-stream"),
]

for fname, mime in files_to_download:
    with open(f"/kaggle/working/{fname}", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    html = f'<a download="{fname}" href="data:{mime};base64,{b64}" style="margin-right:16px;font-size:14px">⬇ {fname}</a>'
    display(HTML(html))

print(f"\n{'='*55}")
print(f"  HOÀN TẤT — Tải 3 file trên về máy")
print(f"  Sau đó chạy: python verify_model.py")
print(f"{'='*55}")