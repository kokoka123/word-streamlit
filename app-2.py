"""
app.py — Word Frequency Analyzer สำหรับนักแปล
สร้างด้วย Streamlit | UI ภาษาไทย
features: top words, POS tagging, theme switcher 7 สี, download CSV
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from collections import Counter

# ─────────────────────────────────────────────
# 2. PAGE CONFIG (ต้องมาก่อน st อื่นทั้งหมด)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="วิเคราะห์คำศัพท์ | สำหรับนักแปล",
    page_icon="📖",
    layout="wide",
)

# ─────────────────────────────────────────────
# 3. THEME SYSTEM — 7 ธีมสี
#    เก็บใน session_state เพื่อให้คงอยู่ข้ามการ rerun
# ─────────────────────────────────────────────

# นิยามธีมสี: (primary_dark, primary_mid, accent, bg_start, bg_end, shadow_rgba)
THEMES = {
    "🔴 แดง":    ("#7f1d1d", "#dc2626", "#ef4444", "#fff5f5", "#ffe4e4", "220,38,38"),
    "🟡 เหลือง": ("#713f12", "#d97706", "#f59e0b", "#fffbeb", "#fef3c7", "217,119,6"),
    "🔵 ฟ้า":    ("#1e3a5f", "#2563eb", "#3b82f6", "#eff6ff", "#dbeafe", "37,99,235"),
    "🩷 ชมพู":   ("#831843", "#db2777", "#ec4899", "#fdf2f8", "#fce7f3", "219,39,119"),
    "🟤 น้ำตาล": ("#3b1c0a", "#92400e", "#b45309", "#fdf8f0", "#fef3c7", "146,64,14"),
    "🟢 เขียว":  ("#14532d", "#16a34a", "#22c55e", "#f0fdf4", "#dcfce7", "22,163,74"),
    "🟣 ม่วง":   ("#2e1065", "#7c3aed", "#8b5cf6", "#faf5ff", "#ede9fe", "124,58,237"),
}

# ค่าเริ่มต้น
if "theme_name" not in st.session_state:
    st.session_state.theme_name = "🔵 ฟ้า"

theme = THEMES[st.session_state.theme_name]
T_DARK, T_MID, T_ACCENT, BG_START, BG_END, SHADOW = theme


# ─────────────────────────────────────────────
# 4. DYNAMIC CSS — ปรับสีตามธีมที่เลือก
# ─────────────────────────────────────────────
def inject_css(dark, mid, accent, bg_start, bg_end, shadow):
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&family=Noto+Serif+Thai:wght@400;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Sarabun', sans-serif;
        }}

        .stApp {{
            background: linear-gradient(135deg, {bg_start} 0%, {bg_end} 100%);
        }}

        .main-header {{
            background: linear-gradient(90deg, {dark}, {mid}, {accent});
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px rgba({shadow}, 0.28);
        }}
        .main-header h1 {{
            color: white;
            font-family: 'Noto Serif Thai', serif;
            font-size: 2rem;
            margin: 0;
        }}
        .main-header p {{
            color: rgba(255,255,255,0.85);
            margin: 0.4rem 0 0 0;
            font-size: 1rem;
        }}

        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.07);
            border-left: 4px solid {mid};
            margin-bottom: 1rem;
        }}
        .stat-card .label {{
            font-size: 0.82rem;
            color: #888;
            margin-bottom: 0.2rem;
        }}
        .stat-card .value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {dark};
        }}

        .section-title {{
            font-family: 'Noto Serif Thai', serif;
            font-size: 1.2rem;
            font-weight: 700;
            color: {dark};
            border-bottom: 2px solid {mid};
            padding-bottom: 0.4rem;
            margin-bottom: 1.2rem;
        }}

        .stDownloadButton > button {{
            background: linear-gradient(90deg, {mid}, {dark}) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 2rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            font-family: 'Sarabun', sans-serif !important;
            box-shadow: 0 4px 12px rgba({shadow}, 0.35) !important;
            transition: all 0.2s ease !important;
        }}
        .stDownloadButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba({shadow}, 0.45) !important;
        }}

        [data-testid="stFileUploader"] {{
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        }}

        [data-testid="stDataFrame"] {{
            border-radius: 10px;
            overflow: hidden;
        }}

        .pos-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            margin: 2px 3px;
            color: white;
        }}

        [data-testid="stSidebar"] {{
            background: white;
        }}
    </style>
    """, unsafe_allow_html=True)

inject_css(T_DARK, T_MID, T_ACCENT, BG_START, BG_END, SHADOW)


# ─────────────────────────────────────────────
# 5. STOPWORDS ภาษาอังกฤษ (ชุดครบถ้วน)
# ─────────────────────────────────────────────
ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "that", "this", "these", "those", "it", "its", "he", "she",
    "they", "we", "you", "i", "me", "him", "her", "them", "us", "my",
    "your", "his", "our", "their", "who", "which", "what",
    "when", "where", "how", "why", "all", "any", "some", "such", "only",
    "own", "same", "than", "too", "very", "just", "more", "also", "into",
    "about", "up", "out", "over", "then", "there", "here", "s",
    "t", "don", "doesn", "didn", "won", "isn", "aren", "wasn", "weren",
    "hasn", "haven", "hadn", "wouldn", "couldn", "shouldn", "re", "ve",
    "ll", "m", "d", "am", "get", "got", "like", "even", "back",
    "well", "still", "way", "make", "made", "take", "took", "go", "went",
    "come", "came", "see", "saw", "know", "knew", "think", "thought",
    "one", "two", "three", "first", "last", "new", "old", "other",
    "after", "before", "between", "through", "during", "without", "against",
    "while", "now", "then", "once", "again", "further", "few",
    "most", "another", "much", "many", "every", "must", "need",
    "across", "along", "around", "near", "off", "per", "since", "though",
    "upon", "within", "among", "under", "above", "below", "beside", "next",
    "however", "although", "because", "therefore", "thus", "hence",
    "whether", "whereas", "despite", "unless", "until",
}


# ─────────────────────────────────────────────
# 6. POS TAGGING — ระบุ Part of Speech
# ─────────────────────────────────────────────

# แผนที่ POS tag → (ชื่อภาษาไทย, สีบัดจ์)
POS_LABEL_TH = {
    "NN":   ("คำนาม",                        "#0369a1"),
    "NNS":  ("คำนาม (พหูพจน์)",              "#0369a1"),
    "NNP":  ("ชื่อเฉพาะ",                    "#0c4a6e"),
    "NNPS": ("ชื่อเฉพาะ (พหูพจน์)",          "#0c4a6e"),
    "VB":   ("คำกริยา",                       "#15803d"),
    "VBD":  ("คำกริยา (อดีต)",               "#15803d"),
    "VBG":  ("คำกริยา (-ing)",               "#15803d"),
    "VBN":  ("คำกริยา (past participle)",    "#15803d"),
    "VBP":  ("คำกริยา (ปัจจุบัน)",           "#15803d"),
    "VBZ":  ("คำกริยา (he/she/it)",          "#15803d"),
    "JJ":   ("คำคุณศัพท์",                   "#b45309"),
    "JJR":  ("คำคุณศัพท์ (เปรียบเทียบ)",    "#b45309"),
    "JJS":  ("คำคุณศัพท์ (ขั้นสุด)",        "#b45309"),
    "RB":   ("คำวิเศษณ์",                    "#7c3aed"),
    "RBR":  ("คำวิเศษณ์ (เปรียบเทียบ)",     "#7c3aed"),
    "RBS":  ("คำวิเศษณ์ (ขั้นสุด)",         "#7c3aed"),
    "IN":   ("คำบุพบท",                      "#9f1239"),
    "DT":   ("Article/Determiner",           "#475569"),
    "CC":   ("คำสันธาน",                     "#64748b"),
    "PRP":  ("สรรพนาม",                      "#0891b2"),
    "PRP$": ("สรรพนาม (เจ้าของ)",           "#0891b2"),
    "WP":   ("คำถาม (who/what)",             "#0891b2"),
    "WDT":  ("คำถาม (which/that)",           "#0891b2"),
    "WRB":  ("คำถาม (where/when/how)",       "#7c3aed"),
    "MD":   ("Modal verb",                   "#065f46"),
    "CD":   ("ตัวเลข",                       "#92400e"),
    "EX":   ("Existential (there is/are)",   "#6b7280"),
    "FW":   ("คำต่างประเทศ",                 "#6b7280"),
    "UH":   ("คำอุทาน",                      "#be185d"),
    "RP":   ("Particle",                     "#6b7280"),
    "TO":   ("to (infinitive marker)",       "#6b7280"),
    "PDT":  ("Pre-determiner",               "#6b7280"),
    "POS":  ("Possessive ('s)",              "#6b7280"),
    "SYM":  ("สัญลักษณ์",                   "#6b7280"),
    "LS":   ("List marker",                  "#6b7280"),
}

# กลุ่ม POS สำหรับ filter ใน sidebar
POS_GROUPS = {
    "ทั้งหมด":                    None,
    "คำนาม (Noun)":               ["NN", "NNS", "NNP", "NNPS"],
    "คำกริยา (Verb)":             ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    "คำคุณศัพท์ (Adjective)":    ["JJ", "JJR", "JJS"],
    "คำวิเศษณ์ (Adverb)":        ["RB", "RBR", "RBS", "WRB"],
    "คำสรรพนาม (Pronoun)":       ["PRP", "PRP$", "WP", "WDT"],
}

# กลุ่มรวมสำหรับ pie chart
POS_PIE_GROUP = {
    "NN": "คำนาม", "NNS": "คำนาม", "NNP": "ชื่อเฉพาะ", "NNPS": "ชื่อเฉพาะ",
    "VB": "คำกริยา", "VBD": "คำกริยา", "VBG": "คำกริยา",
    "VBN": "คำกริยา", "VBP": "คำกริยา", "VBZ": "คำกริยา",
    "JJ": "คำคุณศัพท์", "JJR": "คำคุณศัพท์", "JJS": "คำคุณศัพท์",
    "RB": "คำวิเศษณ์", "RBR": "คำวิเศษณ์", "RBS": "คำวิเศษณ์", "WRB": "คำวิเศษณ์",
    "PRP": "สรรพนาม", "PRP$": "สรรพนาม", "WP": "สรรพนาม", "WDT": "สรรพนาม",
}
PIE_COLORS = {
    "คำนาม": "#0369a1", "ชื่อเฉพาะ": "#0c4a6e",
    "คำกริยา": "#15803d", "คำคุณศัพท์": "#b45309",
    "คำวิเศษณ์": "#7c3aed", "สรรพนาม": "#0891b2", "อื่นๆ": "#94a3b8",
}


def get_pos_tag(word: str) -> tuple:
    """
    ระบุ POS tag ของคำ
    ลำดับ: 1) NLTK tagger (แม่นยำ)  2) Rule-based suffix (fallback)
    คืนค่า: (tag_code, label_thai)
    """
    try:
        import nltk
        for res in ["averaged_perceptron_tagger_eng", "punkt_tab"]:
            try:
                nltk.data.find(f"taggers/{res}")
            except LookupError:
                try:
                    nltk.download(res, quiet=True)
                except Exception:
                    pass
        tokens = nltk.word_tokenize(word)
        if tokens:
            tag = nltk.pos_tag(tokens)[0][1]
            info = POS_LABEL_TH.get(tag, ("คำอื่นๆ", "#6b7280"))
            return tag, info[0]
    except Exception:
        pass

    # ─── Fallback rule-based ───
    w = word.lower()
    if word[0].isupper() and len(word) > 1:
        return "NNP", "ชื่อเฉพาะ"
    if w.endswith(("ing",)):
        return "VBG", "คำกริยา (-ing)"
    if w.endswith(("ize", "ise", "ify", "ate")):
        return "VB", "คำกริยา"
    if w.endswith("ed"):
        return "VBD", "คำกริยา (อดีต)"
    if w.endswith(("tion", "sion", "ness", "ment", "ity", "ship",
                    "hood", "ance", "ence", "age", "ure", "ism")):
        return "NN", "คำนาม"
    if w.endswith("s") and len(w) > 3:
        return "NNS", "คำนาม (พหูพจน์)"
    if w.endswith(("ful", "less", "ous", "ive", "al", "ic", "ish", "able", "ible")):
        return "JJ", "คำคุณศัพท์"
    if w.endswith("ly"):
        return "RB", "คำวิเศษณ์"
    return "NN", "คำนาม"


@st.cache_data(show_spinner=False)
def tag_words_cached(words_tuple: tuple) -> pd.DataFrame:
    """
    วิเคราะห์ POS สำหรับคำที่กำหนด (cached เพื่อความเร็ว)
    คืนค่า DataFrame: คำ, POS Tag, ประเภทคำ (ภาษาไทย), ความถี่
    """
    word_count = Counter(words_tuple)
    results = []
    for word, count in word_count.items():
        tag, label = get_pos_tag(word)
        results.append({
            "คำ": word,
            "POS Tag": tag,
            "ประเภทคำ (ภาษาไทย)": label,
            "ความถี่": count,
        })
    df = pd.DataFrame(results).sort_values("ความถี่", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# ─────────────────────────────────────────────
# 7. FUNCTIONS — อ่านไฟล์ & วิเคราะห์คำ
# ─────────────────────────────────────────────

def read_txt(file) -> str:
    """อ่านไฟล์ .txt รองรับหลาย encoding"""
    raw = file.read()
    for enc in ("utf-8", "utf-8-sig", "latin-1", "tis-620"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("latin-1", errors="replace")


def read_docx(file) -> str:
    """อ่านไฟล์ .docx รวม paragraph ทั้งหมด"""
    try:
        from docx import Document
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except ImportError:
        st.error("❌ กรุณาติดตั้ง python-docx: `pip install python-docx`")
        return ""


def extract_words(text: str, use_stopwords: bool, extra_stopwords: set) -> list:
    """แยกคำภาษาอังกฤษ กรอง stopwords"""
    words = re.findall(r"[a-zA-Z']+", text)
    words = [w.strip("'").lower() for w in words if len(w.strip("'")) > 1]
    if use_stopwords:
        combined = ENGLISH_STOPWORDS | extra_stopwords
        words = [w for w in words if w not in combined]
    return words


def get_top_words(words: list, top_n: int = 30) -> pd.DataFrame:
    """นับความถี่คำ คืน DataFrame Top N"""
    df = pd.DataFrame(Counter(words).most_common(top_n), columns=["คำ", "ความถี่"])
    df.index += 1
    return df


# ─────────────────────────────────────────────
# 8. CHART FUNCTIONS
# ─────────────────────────────────────────────

def draw_bar_chart(df: pd.DataFrame, top_n: int, dark: str, mid: str, accent: str):
    """วาด horizontal bar chart gradient ตามธีมสี"""
    def h2rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

    c1, c2 = h2rgb(accent), h2rgb(dark)
    n = len(df)
    colors = [
        tuple(c1[i] * (1 - t) + c2[i] * t for i in range(3))
        for t in np.linspace(0, 1, n)
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.4)))
    fig.patch.set_facecolor("#fafbff")
    ax.set_facecolor("#fafbff")
    bars = ax.barh(range(n - 1, -1, -1), df["ความถี่"],
                   color=colors, edgecolor="white", linewidth=0.8, height=0.72)

    max_val = df["ความถี่"].max()
    for bar, val in zip(bars, df["ความถี่"]):
        ax.text(bar.get_width() + max_val * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left",
                fontsize=9, color="#333", fontweight="bold")

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["คำ"].tolist()[::-1], fontsize=10.5)
    ax.set_xlabel("ความถี่ (ครั้ง)", fontsize=11, color="#555", labelpad=10)
    ax.set_title(f"Top {top_n} คำที่ใช้บ่อยที่สุด",
                 fontsize=14, fontweight="bold", color=dark, pad=15)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#ddd")
    ax.spines["bottom"].set_color("#ddd")
    ax.tick_params(colors="#555")
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_val * 1.14)
    plt.tight_layout()
    return fig


def draw_pos_pie(df_pos: pd.DataFrame, dark: str):
    """วาด pie chart สัดส่วนประเภทคำ"""
    df_pos = df_pos.copy()
    df_pos["กลุ่ม"] = df_pos["POS Tag"].map(lambda t: POS_PIE_GROUP.get(t, "อื่นๆ"))
    grouped = df_pos.groupby("กลุ่ม")["ความถี่"].sum().sort_values(ascending=False)
    labels = grouped.index.tolist()
    sizes  = grouped.values.tolist()
    clrs   = [PIE_COLORS.get(l, "#94a3b8") for l in labels]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#fafbff")
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=clrs, autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=9),
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(8)
    ax.set_title("สัดส่วนประเภทคำ", fontsize=12, fontweight="bold", color=dark, pad=10)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 9. SIDEBAR — ตั้งค่า
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ ตั้งค่าการวิเคราะห์")
    st.markdown("---")

    top_n = st.slider("📊 จำนวนคำที่แสดง (Top N)",
                      min_value=5, max_value=50, value=30, step=5)

    use_stopwords = st.toggle("🚫 ตัด Stopwords ภาษาอังกฤษ", value=True)

    extra_input = st.text_area(
        "➕ เพิ่ม Stopwords เอง (คั่นด้วยลูกน้ำ)",
        placeholder="เช่น: said, went, came",
        height=90,
    )
    extra_stopwords = {w.strip().lower() for w in extra_input.split(",") if w.strip()}

    st.markdown("---")
    st.markdown("### 🔤 กรองประเภทคำ (POS)")
    pos_filter = st.selectbox("แสดงเฉพาะ:", list(POS_GROUPS.keys()),
                               index=0, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<small style='color:#aaa;'>📖 Word Frequency Analyzer<br>สำหรับนักแปลและนักภาษาศาสตร์</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# 10. THEME SWITCHER — ปุ่ม 7 สี
# ─────────────────────────────────────────────

# สีพื้นหลัง/ตัวอักษร ของปุ่มแต่ละธีม
THEME_BTN_STYLE = {
    "🔴 แดง":    ("#fee2e2", "#991b1b", "#dc2626"),
    "🟡 เหลือง": ("#fef9c3", "#713f12", "#d97706"),
    "🔵 ฟ้า":    ("#dbeafe", "#1e3a5f", "#2563eb"),
    "🩷 ชมพู":   ("#fce7f3", "#831843", "#db2777"),
    "🟤 น้ำตาล": ("#fef3c7", "#3b1c0a", "#92400e"),
    "🟢 เขียว":  ("#dcfce7", "#14532d", "#16a34a"),
    "🟣 ม่วง":   ("#ede9fe", "#2e1065", "#7c3aed"),
}

st.markdown("#### 🎨 เลือกธีมสีเว็บไซต์")
cols = st.columns(7)
for i, t_name in enumerate(THEMES.keys()):
    bg, fg, border = THEME_BTN_STYLE[t_name]
    is_active = (st.session_state.theme_name == t_name)
    outline = f"3px solid {border}" if is_active else f"2px solid {border}55"
    weight  = "800" if is_active else "500"
    shadow  = f"0 3px 10px {border}66" if is_active else "none"

    with cols[i]:
        if st.button(t_name, key=f"theme_{t_name}",
                     help=f"ธีมสี {t_name}", use_container_width=True):
            st.session_state.theme_name = t_name
            st.rerun()

        # CSS override ปุ่มแต่ละธีม (สีพื้น + ขอบถ้าถูกเลือก)
        st.markdown(f"""
        <style>
        div[data-testid="stHorizontalBlock"] > div:nth-child({i+1})
        div[data-testid="stButton"] > button {{
            background: {bg} !important;
            color: {fg} !important;
            border: {outline} !important;
            font-weight: {weight} !important;
            box-shadow: {shadow} !important;
            border-radius: 8px !important;
            font-size: 0.78rem !important;
            padding: 0.35rem 0.1rem !important;
        }}
        </style>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 11. HEADER & UPLOAD
# ─────────────────────────────────────────────

st.markdown(f"""
<div class="main-header">
    <h1>📖 วิเคราะห์ความถี่คำศัพท์</h1>
    <p>เครื่องมือสำหรับนักแปล — วิเคราะห์คำ · ระบุประเภทคำ (POS) · ดาวน์โหลด CSV</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">📁 อัปโหลดไฟล์ข้อความ</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "รองรับไฟล์ .txt และ .docx",
    type=["txt", "docx"],
    label_visibility="collapsed",
)


# ─────────────────────────────────────────────
# 12. PROCESSING — เมื่อมีไฟล์
# ─────────────────────────────────────────────

if uploaded_file is not None:

    # อ่านไฟล์
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext == "txt":
        text = read_txt(uploaded_file)
    elif file_ext == "docx":
        text = read_docx(uploaded_file)
    else:
        st.error("❌ ไม่รองรับไฟล์ประเภทนี้")
        st.stop()

    if not text.strip():
        st.warning("⚠️ ไม่พบเนื้อหาในไฟล์")
        st.stop()

    # แยกและนับคำ
    words = extract_words(text, use_stopwords, extra_stopwords)
    if not words:
        st.warning("⚠️ ไม่พบคำที่วิเคราะห์ได้หลังกรอง Stopwords")
        st.stop()

    total_words  = len(words)
    unique_words = len(set(words))
    df_top       = get_top_words(words, top_n=top_n)

    # ─── Stat Cards ──────────────────────────
    st.markdown('<div class="section-title">📊 สรุปผลการวิเคราะห์</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    top_word = df_top.iloc[0]["คำ"] if not df_top.empty else "-"
    top_freq = df_top.iloc[0]["ความถี่"] if not df_top.empty else 0

    for col, lbl, val in [
        (c1, "📄 ชื่อไฟล์",              f'<div style="font-size:0.95rem;">{uploaded_file.name}</div>'),
        (c2, "📝 คำทั้งหมด (หลังกรอง)", f"{total_words:,}"),
        (c3, "🔤 คำไม่ซ้ำกัน",          f"{unique_words:,}"),
        (c4, "🏆 คำที่ใช้บ่อยสุด",      f'"{top_word}" ({top_freq:,})'),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="label">{lbl}</div>
            <div class="value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── 2 Tab: ความถี่ | POS ────────────────
    tab1, tab2 = st.tabs(["📊 ความถี่คำ (Top Words)", "🔤 วิเคราะห์ประเภทคำ (POS)"])

    # ══════ TAB 1: TOP WORDS ══════════════════
    with tab1:
        col_chart, col_table = st.columns([3, 2], gap="large")

        with col_chart:
            st.markdown(f'<div class="section-title">📊 Bar Chart — Top {top_n} คำ</div>',
                        unsafe_allow_html=True)
            fig = draw_bar_chart(df_top, top_n, T_DARK, T_MID, T_ACCENT)
            st.pyplot(fig, use_container_width=True)

        with col_table:
            st.markdown(f'<div class="section-title">📋 ตาราง Top {top_n} คำ</div>',
                        unsafe_allow_html=True)
            st.dataframe(df_top, use_container_width=True,
                         height=min(600, top_n * 38 + 40))

        # Download Tab 1
        st.markdown("---")
        st.markdown('<div class="section-title">💾 ดาวน์โหลดผลลัพธ์</div>', unsafe_allow_html=True)
        fname = uploaded_file.name.rsplit(".", 1)[0] + "_word_frequency.csv"
        st.download_button(
            label=f"⬇️ ดาวน์โหลด Top {top_n} คำ เป็น CSV",
            data=df_top.to_csv(index=True).encode("utf-8-sig"),
            file_name=fname, mime="text/csv",
        )
        st.caption(f"💡 ไฟล์: `{fname}` — ใช้งานได้ใน Excel และ Google Sheets")

    # ══════ TAB 2: POS ANALYSIS ═══════════════
    with tab2:
        st.markdown('<div class="section-title">🔤 การวิเคราะห์ประเภทคำ (Part of Speech)</div>',
                    unsafe_allow_html=True)

        st.info(
            "🧠 **วิธีการระบุประเภทคำ:** ใช้ **NLTK averaged perceptron tagger** "
            "(ถ้าติดตั้งไว้) หรือ **rule-based suffix pattern** เป็น fallback\n\n"
            "💡 ติดตั้ง NLTK เพื่อความแม่นยำสูงสุด: `pip install nltk`"
        )

        with st.spinner("⏳ กำลังวิเคราะห์ประเภทคำ..."):
            # วิเคราะห์เฉพาะคำที่อยู่ใน Top Words (เร็วกว่าวิเคราะห์ทั้งหมด)
            top_word_set = set(df_top["คำ"].tolist())
            filtered_words = tuple(w for w in words if w in top_word_set)
            df_pos = tag_words_cached(filtered_words)

        # กรองตาม POS group ที่ผู้ใช้เลือกใน sidebar
        selected_tags = POS_GROUPS[pos_filter]
        df_pos_show = df_pos[df_pos["POS Tag"].isin(selected_tags)].copy() \
            if selected_tags else df_pos.copy()

        if df_pos_show.empty:
            st.warning("ไม่พบคำในหมวดหมู่ที่เลือกใน Top Words")
        else:
            col_tbl, col_pie = st.columns([3, 2], gap="large")

            # ── ตาราง POS (HTML สวยงาม) ──
            with col_tbl:
                st.markdown(f"**พบ {len(df_pos_show)} คำ** | หมวด: {pos_filter}")

                row_htmls = []
                for idx, (_, row) in enumerate(df_pos_show.iterrows()):
                    tag   = row["POS Tag"]
                    label = row["ประเภทคำ (ภาษาไทย)"]
                    word  = row["คำ"]
                    freq  = row["ความถี่"]
                    color = POS_LABEL_TH.get(tag, ("", "#6b7280"))[1]
                    bg    = "#ffffff" if idx % 2 == 0 else "#f8faff"
                    row_htmls.append(f"""
                    <tr style="background:{bg};">
                        <td style="padding:8px 14px; font-weight:600; color:#1e293b;">{word}</td>
                        <td style="padding:8px 14px;">
                            <span class="pos-badge" style="background:{color};">{tag}</span>
                        </td>
                        <td style="padding:8px 14px; color:#475569; font-size:0.9rem;">{label}</td>
                        <td style="padding:8px 14px; font-weight:700; color:{T_DARK}; text-align:right;">{freq:,}</td>
                    </tr>""")

                table_html = f"""
                <div style="overflow-y:auto; max-height:500px; border-radius:12px;
                            box-shadow:0 4px 16px rgba(0,0,0,0.08);">
                <table style="width:100%; border-collapse:collapse; background:white;
                              font-family:'Sarabun',sans-serif;">
                    <thead>
                        <tr style="background:{T_MID}; color:white; position:sticky; top:0; z-index:1;">
                            <th style="padding:11px 14px; text-align:left;">คำ</th>
                            <th style="padding:11px 14px; text-align:left;">POS Tag</th>
                            <th style="padding:11px 14px; text-align:left;">ประเภทคำ (ไทย)</th>
                            <th style="padding:11px 14px; text-align:right;">ความถี่</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(row_htmls)}</tbody>
                </table>
                </div>"""
                st.markdown(table_html, unsafe_allow_html=True)

            # ── Pie chart ──
            with col_pie:
                st.markdown("**สัดส่วนประเภทคำ**")
                fig_pie = draw_pos_pie(df_pos, T_DARK)
                st.pyplot(fig_pie, use_container_width=True)

            # ── POS Legend ──
            st.markdown("---")
            st.markdown("**📚 คำอธิบาย POS Tag ที่พบในผลลัพธ์:**")
            seen, badges = set(), ""
            for tag in df_pos_show["POS Tag"].unique():
                if tag in seen:
                    continue
                seen.add(tag)
                info  = POS_LABEL_TH.get(tag, ("คำอื่นๆ", "#6b7280"))
                badges += (f'<span class="pos-badge" style="background:{info[1]};">'
                           f'{tag} = {info[0]}</span> ')
            st.markdown(f'<div style="line-height:2.4;">{badges}</div>', unsafe_allow_html=True)

            # ── Download POS CSV ──
            st.markdown("<br>", unsafe_allow_html=True)
            fname_pos = uploaded_file.name.rsplit(".", 1)[0] + "_pos_analysis.csv"
            st.download_button(
                label="⬇️ ดาวน์โหลดผลวิเคราะห์ POS เป็น CSV",
                data=df_pos_show.to_csv(index=True).encode("utf-8-sig"),
                file_name=fname_pos, mime="text/csv",
            )
            st.caption(f"💡 ไฟล์: `{fname_pos}` — รวมคอลัมน์ POS Tag และประเภทคำภาษาไทย")


# ─────────────────────────────────────────────
# 13. EMPTY STATE — ยังไม่ได้อัปโหลด
# ─────────────────────────────────────────────
else:
    icons = [("🔍","วิเคราะห์คำทันที"), ("🚫","ตัด Stopwords"),
             ("🔤","ระบุ POS"), ("📊","Bar Chart"), ("💾","Export CSV")]
    icon_html = "".join(
        f'<div style="text-align:center;"><div style="font-size:1.8rem;">{ic}</div>'
        f'<div style="font-size:0.88rem;color:#666;margin-top:0.3rem;">{lb}</div></div>'
        for ic, lb in icons
    )
    st.markdown(f"""
    <div style="text-align:center; padding:4rem 2rem; background:white;
                border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.06); margin-top:1rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">📂</div>
        <h3 style="color:{T_DARK}; font-family:'Noto Serif Thai',serif;">อัปโหลดไฟล์เพื่อเริ่มต้น</h3>
        <p style="color:#888; font-size:1rem; max-width:450px; margin:0 auto;">
            รองรับไฟล์ <strong>.txt</strong> และ <strong>.docx</strong><br>
            วิเคราะห์ความถี่คำ · ระบุประเภทคำ (POS) · Export CSV
        </p>
        <div style="display:flex; justify-content:center; gap:2.5rem; margin-top:2.2rem; flex-wrap:wrap;">
            {icon_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
