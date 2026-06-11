import streamlit as st
import torch
import torch.nn as nn
import json
import os
import re
import time
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="SocialAI",
    page_icon="◈",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0a;
    color: #e8e8e8;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1100px; margin: auto; }

/* ---- HEADER ---- */
.app-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 2rem;
}
.app-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -1px;
    margin: 0;
}
.app-header p {
    color: #555;
    font-size: 0.82rem;
    margin: 0.4rem 0 0 0;
    font-family: 'IBM Plex Mono', monospace;
}

/* ---- AUTH ---- */
.auth-box {
    background: #111;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 2rem;
    max-width: 420px;
    margin: 3rem auto;
}

/* ---- POST CARD ---- */
.post-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.post-card:hover { border-color: #2a2a2a; }
.post-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.6rem;
}
.avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    background: #1a2a1a;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem;
    font-weight: 600;
    color: #4a9e4a;
    font-family: 'IBM Plex Mono', monospace;
    flex-shrink: 0;
}
.post-author { font-weight: 600; font-size: 0.88rem; color: #e8e8e8; }
.post-time { font-size: 0.72rem; color: #444; font-family: 'IBM Plex Mono', monospace; }
.post-body { font-size: 0.93rem; color: #bbb; line-height: 1.65; margin-bottom: 0.8rem; }
.post-topic-tag {
    display: inline-block;
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    background: #151515;
    border: 1px solid #2a2a2a;
    color: #777;
    padding: 2px 8px;
    border-radius: 3px;
    margin-bottom: 0.6rem;
}

/* ---- COMMENT ---- */
.comment-item {
    background: #0d0d0d;
    border-left: 2px solid #222;
    padding: 0.6rem 0.8rem;
    margin: 0.4rem 0 0.4rem 1rem;
    border-radius: 0 4px 4px 0;
}
.comment-author { font-size: 0.78rem; font-weight: 600; color: #666; margin-bottom: 0.2rem; }
.comment-body { font-size: 0.87rem; color: #aaa; line-height: 1.5; }

/* ---- ADMIN DASHBOARD ---- */
.dash-header {
    padding: 1.2rem 0 1rem 0;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 1.5rem;
}
.dash-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #fff;
    margin: 0;
}
.dash-subtitle {
    font-size: 0.75rem;
    color: #444;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.3rem;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #fff;
}
.stat-label {
    font-size: 0.72rem;
    color: #555;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
}

.user-row {
    background: #111;
    border: 1px solid #1a1a1a;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.2s;
}
.user-row:hover { border-color: #252525; }
.user-row-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.7rem;
}
.user-info { display: flex; align-items: center; gap: 0.6rem; }
.user-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    color: #e8e8e8;
}
.user-meta { font-size: 0.73rem; color: #555; font-family: 'IBM Plex Mono', monospace; }
.user-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 12px;
    border: 1px solid;
}
.badge-tech    { color: #5ba8d8; border-color: #1a3a5f; background: #0d1e30; }
.badge-sports  { color: #e07b5a; border-color: #5f2a1a; background: #2e1610; }
.badge-finance { color: #c8a84b; border-color: #5f4a1a; background: #2e2410; }
.badge-gaming  { color: #9e6bdb; border-color: #3a1a5f; background: #1e1030; }
.badge-unknown { color: #666; border-color: #333; background: #111; }

.mini-bar-row { margin-bottom: 0.35rem; }
.mini-bar-label {
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #666;
    margin-bottom: 2px;
    display: flex;
    justify-content: space-between;
}
.mini-bar-track { background: #1a1a1a; border-radius: 2px; height: 4px; width: 100%; }
.mini-bar-fill-tech    { height: 4px; border-radius: 2px; background: #5ba8d8; }
.mini-bar-fill-sports  { height: 4px; border-radius: 2px; background: #e07b5a; }
.mini-bar-fill-finance { height: 4px; border-radius: 2px; background: #c8a84b; }
.mini-bar-fill-gaming  { height: 4px; border-radius: 2px; background: #9e6bdb; }

.refresh-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #3a3a3a;
    text-align: right;
    margin-bottom: 1rem;
}
.refresh-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: #2a5a2a;
    border-radius: 50%;
    margin-right: 5px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ---- PROFILE ---- */
.profile-badge {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.profile-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem; font-weight: 600; color: #fff; }
.profile-role { font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace; margin-top: 0.25rem; }
.role-admin { color: #c8a84b; }
.role-user  { color: #555; }

/* ---- MISC ---- */
.section-divider { border: none; border-top: 1px solid #151515; margin: 1.2rem 0; }
.success-msg { color: #4a9e4a; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; padding: 0.4rem 0; }
.error-msg   { color: #c0392b; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; padding: 0.4rem 0; }
.info-msg    { color: #555; font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; padding: 0.2rem 0; }

.stButton > button {
    background: #151515 !important;
    color: #e8e8e8 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 5px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { border-color: #444 !important; background: #1e1e1e !important; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0d0d0d !important;
    color: #e8e8e8 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 5px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus { border-color: #333 !important; box-shadow: none !important; }
.stSelectbox > div > div { background: #0d0d0d !important; border: 1px solid #1e1e1e !important; color: #e8e8e8 !important; }
label { color: #666 !important; font-size: 0.8rem !important; }
.stTabs [data-baseweb="tab"] { color: #555 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important; }
.stTabs [aria-selected="true"] { color: #e8e8e8 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.dropout(hidden[-1])
        return self.fc(out)

# ============================================================
# CONSTANTS
# ============================================================
VOCAB_SIZE  = 10000
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
NUM_CLASSES = 4
MAX_LEN     = 50

try:
    with open("label_map.json", "r", encoding="utf-8") as f:
        _lm = json.load(f)
    LABEL_MAP = {int(k): v for k, v in _lm.items()}
except FileNotFoundError:
    LABEL_MAP = {0: "Tech", 1: "Sports", 2: "Finance", 3: "Gaming"}

LABEL_EMOJI = {"Tech": "💻", "Sports": "⚽", "Finance": "📈", "Gaming": "🎮"}
BADGE_CLASS = {"Tech": "badge-tech", "Sports": "badge-sports",
               "Finance": "badge-finance", "Gaming": "badge-gaming"}
BAR_CLASS   = {"Tech": "mini-bar-fill-tech", "Sports": "mini-bar-fill-sports",
               "Finance": "mini-bar-fill-finance", "Gaming": "mini-bar-fill-gaming"}

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

SAMPLE_POSTS = [
    {"id": 1, "author": "admin", "topic": "Sports",
     "body": "Trận chung kết Champions League tối qua thật kịch tính! Bàn thắng phút 90 khiến cả sân vỡ òa. Ai xem không?",
     "time": "2h trước"},
    {"id": 2, "author": "admin", "topic": "Tech",
     "body": "AI đang thay đổi ngành lập trình. Các IDE tích hợp LLM đang trở nên phổ biến. Bạn nghĩ developer sẽ bị thay thế không?",
     "time": "5h trước"},
    {"id": 3, "author": "admin", "topic": "Finance",
     "body": "Bitcoin vừa chạm mốc ATH mới. Thị trường crypto đang rất sôi động. Anh em có đang hold coin không?",
     "time": "8h trước"},
    {"id": 4, "author": "admin", "topic": "Gaming",
     "body": "GTA VI trailer vừa drop! Graphics trông điên thật. Release date cuối năm nay — hype quá!",
     "time": "1 ngày trước"},
]

# ============================================================
# DATA PERSISTENCE
# ============================================================
USERS_FILE    = "users.json"
COMMENTS_FILE = "comments.json"

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def init_admin():
    """Tạo tài khoản admin mặc định nếu chưa tồn tại."""
    users = load_json(USERS_FILE, {})
    if "admin" not in users:
        users["admin"] = {
            "password": "admin123",
            "role": "admin",
            "comments": []
        }
        save_json(USERS_FILE, users)

# ============================================================
# LOAD MODEL + VOCAB
# ============================================================
@st.cache_resource
def load_model_and_vocab():
    if not os.path.exists("word2idx.json"):
        return None, None
    with open("word2idx.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    if not os.path.exists("best_model.pt"):
        return None, word2idx
    model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES)
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
    model.eval()
    return model, word2idx

# ============================================================
# INFERENCE
# ============================================================
def tokenize(text):
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text.lower())
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 2]

def encode_text(text, word2idx, max_len=MAX_LEN):
    tokens = tokenize(text)[:max_len]
    ids    = [word2idx.get(t, 1) for t in tokens]
    ids   += [0] * (max_len - len(ids))
    return ids

def predict(texts, model, word2idx):
    """
    Predict mỗi comment riêng lẻ → lấy trung bình xác suất → argmax.
    Phản ánh đúng tần suất sở thích thực tế của user.
    """
    all_probs = []
    for text in texts:
        ids    = encode_text(text, word2idx)
        tensor = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)[0].tolist()
        all_probs.append(probs)

    # Trung bình xác suất qua tất cả comment
    avg_probs = [
        sum(p[i] for p in all_probs) / len(all_probs)
        for i in range(NUM_CLASSES)
    ]
    pred_idx = avg_probs.index(max(avg_probs))
    return LABEL_MAP[pred_idx], avg_probs

# ============================================================
# INIT
# ============================================================
init_admin()

# ============================================================
# SESSION STATE
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""

# ============================================================
# LOAD RESOURCES
# ============================================================
model, word2idx = load_model_and_vocab()
users_db    = load_json(USERS_FILE, {})
comments_db = load_json(COMMENTS_FILE, {str(p["id"]): [] for p in SAMPLE_POSTS})

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <h1>◈ SocialAI</h1>
    <p>social network · behavioral analytics platform</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# AUTH
# ============================================================
if not st.session_state.logged_in:
    st.markdown('<div class="auth-box">', unsafe_allow_html=True)
    tab_login, tab_register = st.tabs(["Đăng nhập", "Đăng ký"])

    with tab_login:
        st.markdown("")
        uname = st.text_input("Tên đăng nhập", key="login_user")
        pwd   = st.text_input("Mật khẩu", type="password", key="login_pwd")
        if st.button("Đăng nhập", key="btn_login"):
            users_db = load_json(USERS_FILE, {})
            if uname in users_db and users_db[uname]["password"] == pwd:
                st.session_state.logged_in = True
                st.session_state.username  = uname
                st.session_state.role      = users_db[uname].get("role", "user")
                st.rerun()
            else:
                st.markdown('<p class="error-msg">❌ Sai tên đăng nhập hoặc mật khẩu</p>',
                            unsafe_allow_html=True)

    with tab_register:
        st.markdown("")
        new_user = st.text_input("Tên đăng nhập", key="reg_user")
        new_pwd  = st.text_input("Mật khẩu", type="password", key="reg_pwd")
        new_pwd2 = st.text_input("Nhập lại mật khẩu", type="password", key="reg_pwd2")
        if st.button("Đăng ký", key="btn_register"):
            users_db = load_json(USERS_FILE, {})
            if not new_user or not new_pwd:
                st.markdown('<p class="error-msg">❌ Vui lòng điền đầy đủ thông tin</p>',
                            unsafe_allow_html=True)
            elif new_user in users_db:
                st.markdown('<p class="error-msg">❌ Tên đăng nhập đã tồn tại</p>',
                            unsafe_allow_html=True)
            elif new_pwd != new_pwd2:
                st.markdown('<p class="error-msg">❌ Mật khẩu không khớp</p>',
                            unsafe_allow_html=True)
            elif len(new_user) < 3:
                st.markdown('<p class="error-msg">❌ Tên đăng nhập tối thiểu 3 ký tự</p>',
                            unsafe_allow_html=True)
            elif new_user == "admin":
                st.markdown('<p class="error-msg">❌ Tên đăng nhập không hợp lệ</p>',
                            unsafe_allow_html=True)
            else:
                users_db[new_user] = {"password": new_pwd, "role": "user", "comments": []}
                save_json(USERS_FILE, users_db)
                st.markdown('<p class="success-msg">✅ Đăng ký thành công! Chuyển sang đăng nhập.</p>',
                            unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# MAIN APP — ADMIN
# ============================================================
elif st.session_state.role == "admin":
    username = st.session_state.username

    with st.sidebar:
        st.markdown(f"""
        <div class="profile-badge">
            <div class="profile-name">@{username}</div>
            <div class="profile-role role-admin">◈ administrator</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Đăng xuất"):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.role      = ""
            st.rerun()
        st.markdown("---")
        st.markdown('<p class="info-msg">MODEL STATUS</p>', unsafe_allow_html=True)
        if model is not None:
            st.markdown('<p class="success-msg">✅ LSTM loaded</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="error-msg">❌ Model chưa load</p>', unsafe_allow_html=True)

    # ---- Auto-refresh mỗi 5 giây ----
    refresh_placeholder = st.empty()

    # Reload dữ liệu mới nhất từ file
    users_db    = load_json(USERS_FILE, {})
    comments_db = load_json(COMMENTS_FILE, {str(p["id"]): [] for p in SAMPLE_POSTS})

    # Lọc chỉ user (không tính admin)
    all_users = {k: v for k, v in users_db.items() if v.get("role", "user") == "user"}

    now_str = datetime.now().strftime("%H:%M:%S")
    refresh_placeholder.markdown(
        f'<div class="refresh-badge"><span class="refresh-dot"></span>auto-refresh · cập nhật lúc {now_str}</div>',
        unsafe_allow_html=True
    )

    # ---- Dashboard header ----
    total_users    = len(all_users)
    total_comments = sum(len(v.get("comments", [])) for v in all_users.values())
    active_users   = sum(1 for v in all_users.values() if len(v.get("comments", [])) > 0)

    # Đếm phân loại
    label_counts = {"Tech": 0, "Sports": 0, "Finance": 0, "Gaming": 0, "—": 0}
    for udata in all_users.values():
        coms = udata.get("comments", [])
        if coms and model and word2idx:
            lbl, _ = predict(coms, model, word2idx)
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        else:
            label_counts["—"] += 1

    dominant = max(
        (k for k in label_counts if k != "—"),
        key=lambda k: label_counts[k],
        default="—"
    )

    st.markdown("""
    <div class="dash-header">
        <div class="dash-title">Admin Dashboard</div>
        <div class="dash-subtitle">Phân tích hành vi & sở thích người dùng · real-time</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Stat cards ----
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">{total_users}</div>
            <div class="stat-label">TỔNG USER</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{active_users}</div>
            <div class="stat-label">ĐÃ HOẠT ĐỘNG</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_comments}</div>
            <div class="stat-label">TỔNG COMMENT</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{LABEL_EMOJI.get(dominant, "—")}</div>
            <div class="stat-label">CHỦ ĐỀ PHỔ BIẾN NHẤT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Label distribution bar ----
    if total_users > 0:
        dist_html = '<div style="margin-bottom:1.5rem;">'
        dist_html += '<p class="info-msg" style="margin-bottom:0.6rem;">PHÂN BỐ SỞ THÍCH TOÀN HỆ THỐNG</p>'
        for lbl in ["Tech", "Sports", "Finance", "Gaming"]:
            cnt = label_counts.get(lbl, 0)
            pct = (cnt / total_users * 100) if total_users > 0 else 0
            bar_cls = BAR_CLASS[lbl]
            dist_html += f"""
            <div class="mini-bar-row">
                <div class="mini-bar-label">
                    <span>{LABEL_EMOJI[lbl]} {lbl}</span>
                    <span>{cnt} user · {pct:.1f}%</span>
                </div>
                <div class="mini-bar-track">
                    <div class="{bar_cls}" style="width:{pct}%"></div>
                </div>
            </div>"""
        dist_html += '</div>'
        st.markdown(dist_html, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="info-msg" style="margin-bottom:0.8rem;">CHI TIẾT TỪNG USER</p>',
                unsafe_allow_html=True)

    # ---- Per-user rows ----
    if not all_users:
        st.markdown('<p class="info-msg">Chưa có user nào đăng ký.</p>', unsafe_allow_html=True)
    else:
        for uname, udata in all_users.items():
            coms = udata.get("comments", [])
            num_coms = len(coms)
            initials = uname[:2].upper()

            if coms and model and word2idx:
                lbl, probs = predict(coms, model, word2idx)
                badge_cls  = BADGE_CLASS.get(lbl, "badge-unknown")
                emoji      = LABEL_EMOJI.get(lbl, "?")

                # Build bars HTML trước — tránh lỗi escape khi lồng f-string
                bars_parts = []
                for i, lname in LABEL_MAP.items():
                    pct = round(probs[i] * 100, 2)
                    bc  = BAR_CLASS.get(lname, "mini-bar-fill-tech")
                    bars_parts.append(
                        '<div class="mini-bar-row">'
                        '<div class="mini-bar-label">'
                        '<span>' + LABEL_EMOJI[lname] + ' ' + lname + '</span>'
                        '<span>' + f"{pct:.1f}" + '%</span>'
                        '</div>'
                        '<div class="mini-bar-track">'
                        '<div class="' + bc + '" style="width:' + str(pct) + '%"></div>'
                        '</div>'
                        '</div>'
                    )
                bars_html = "".join(bars_parts)

                # Build full card HTML rồi render 1 lần duy nhất
                card_html = (
                    '<div class="user-row">'
                    '<div class="user-row-header">'
                    '<div class="user-info">'
                    '<div class="avatar">' + initials + '</div>'
                    '<div>'
                    '<div class="user-name">@' + uname + '</div>'
                    '<div class="user-meta">' + str(num_coms) + ' comment</div>'
                    '</div>'
                    '</div>'
                    '<span class="user-badge ' + badge_cls + '">' + emoji + ' ' + lbl + '</span>'
                    '</div>'
                    + bars_html +
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

            else:
                card_html = (
                    '<div class="user-row">'
                    '<div class="user-row-header">'
                    '<div class="user-info">'
                    '<div class="avatar">' + initials + '</div>'
                    '<div>'
                    '<div class="user-name">@' + uname + '</div>'
                    '<div class="user-meta">0 comment · chưa có dữ liệu</div>'
                    '</div>'
                    '</div>'
                    '<span class="user-badge badge-unknown">— chưa phân loại</span>'
                    '</div>'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

    # ---- Auto-refresh mỗi 5 giây ----
    time.sleep(5)
    st.rerun()

# ============================================================
# MAIN APP — USER
# ============================================================
else:
    username = st.session_state.username

    with st.sidebar:
        st.markdown(f"""
        <div class="profile-badge">
            <div class="profile-name">@{username}</div>
            <div class="profile-role role-user">● thành viên</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Đăng xuất"):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.role      = ""
            st.rerun()

    st.markdown(f'<p class="info-msg">Xin chào, <b style="color:#ccc">@{username}</b></p>',
                unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Reload comments mới nhất
    comments_db = load_json(COMMENTS_FILE, {str(p["id"]): [] for p in SAMPLE_POSTS})

    for post in SAMPLE_POSTS:
        pid      = str(post["id"])
        initials = post["author"][:2].upper()

        st.markdown(f"""
        <div class="post-card">
            <div class="post-header">
                <div class="avatar">{initials}</div>
                <div>
                    <div class="post-author">@{post['author']}</div>
                    <div class="post-time">{post['time']}</div>
                </div>
            </div>
            <div class="post-topic-tag">#{post['topic']}</div>
            <div class="post-body">{post['body']}</div>
        </div>
        """, unsafe_allow_html=True)

        post_comments = comments_db.get(pid, [])
        if post_comments:
            for c in post_comments:
                st.markdown(f"""
                <div class="comment-item">
                    <div class="comment-author">@{c['author']} · {c['time']}</div>
                    <div class="comment-body">{c['body']}</div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander(f"💬 Viết comment"):
            comment_text = st.text_area(
                "Nội dung",
                key=f"comment_{pid}",
                placeholder="Nhập comment của bạn...",
                height=80
            )
            if st.button("Đăng", key=f"btn_{pid}"):
                if comment_text.strip():
                    now     = datetime.now().strftime("%H:%M %d/%m")
                    new_cmt = {"author": username, "body": comment_text.strip(), "time": now}
                    comments_db[pid].append(new_cmt)
                    save_json(COMMENTS_FILE, comments_db)

                    # Lưu comment vào profile user để phân tích
                    users_db = load_json(USERS_FILE, {})
                    if username not in users_db:
                        users_db[username] = {"password": "", "role": "user", "comments": []}
                    users_db[username].setdefault("comments", []).append(comment_text.strip())
                    save_json(USERS_FILE, users_db)

                    st.markdown('<p class="success-msg">✅ Đã đăng comment!</p>',
                                unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.markdown('<p class="error-msg">❌ Comment không được để trống</p>',
                                unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)