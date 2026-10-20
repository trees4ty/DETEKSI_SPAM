import time
from jcopml.utils import load_model
import nltk
from nltk.tokenize import word_tokenize
import streamlit as st

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

# ==========================================
# 1. KONFIGURASI HALAMAN (Wajib di paling atas)
# ==========================================
st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==========================================
# 2. CUSTOM CSS & HTML (Tampilan Minimalis & Elegan)
# ==========================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        color: #0F172A;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #64748B;
        font-size: 1rem;
        margin-top: 0;
    }
    div.stButton > button:first-child {
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .spam-card {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 6px solid #EF4444;
    }
    .spam-title {
        color: #991B1B;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .spam-desc {
        color: #B91C1C;
        font-size: 0.9rem;
        margin: 0;
    }
    .ham-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 6px solid #22C55E;
    }
    .ham-title {
        color: #166534;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .ham-desc {
        color: #15803D;
        font-size: 0.9rem;
        margin: 0;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.75rem;
        background-color: rgba(255, 255, 255, 0.8);
        color: #334155;
        border: 1px solid rgba(0,0,0,0.08);
    }
    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 3. FUNGSI LOAD PIPELINE MODEL (Dengan Caching)
# ==========================================
@st.cache_resource
def load_ml_pipeline():
    # Cukup load 1 file pipeline saja (sudah termasuk TfidfVectorizer + LogisticRegression)
    pipeline_model = load_model("model/spam_classifier_model.pkl")
    return pipeline_model


model = load_ml_pipeline()


# ==========================================
# 4. FUNGSI PREDIKSI
# ==========================================
def predict_message(text, pipeline_model):

    clean_text = text.strip()

    # Langsung masukkan list teks mentah ke dalam pipeline!
    # Pipeline otomatis menjalankan step 'prep' (TF-IDF) lalu 'algo' (LogisticRegression)
    prediction = pipeline_model.predict([clean_text])[0]
    probabilities = pipeline_model.predict_proba([clean_text])[0]
    confidence = max(probabilities) * 100

    return prediction, confidence


# ==========================================
# 5. TAMPILAN ANTARMUKA PENGGUNA (UI)
# ==========================================
st.markdown(
    """
    <div class="main-header">
        <h1>Deteksi Pesan Spam</h1>
        <p>Masukkan teks pesan SMS, WhatsApp, atau Email untuk menganalisis indikasi spam.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Inisialisasi state untuk teks input
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Tombol Quick Action
col_ex1, col_ex2, col_clear = st.columns([1, 1, 1])
with col_ex1:
    if st.button("Contoh Spam", use_container_width=True):
        st.session_state.input_text = "SELAMAT! Nomor Anda terpilih mendapatkan hadiah Rp 50 Juta. Klik link bit.ly/hadiah-anda untuk klaim sekarang!"
with col_ex2:
    if st.button("Contoh Pesan Normal", use_container_width=True):
        st.session_state.input_text = "Halo, nanti sore jadi rapat evaluasi proyek jam 4 sore? Tolong kabari kalau sudah di ruangan ya."
with col_clear:
    if st.button("Bersihkan Teks", use_container_width=True):
        st.session_state.input_text = ""

# Gunakan parameter key="input_text" agar otomatis tersinkronisasi dengan st.session_state.input_text
user_input = st.text_area(
    label="Teks Pesan",
    key="input_text",
    height=140,
    placeholder="Ketik atau tempel (paste) pesan yang ingin diperiksa di sini...",
    label_visibility="collapsed",
)

analyze_button = st.button("Analisis Pesan Sekarang")

# ==========================================
# 6. LOGIKA EKSEKUSI & TAMPILAN HASIL
# ==========================================
if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Mohon masukkan teks pesan terlebih dahulu sebelum melakukan analisis.")
    else:
        with st.spinner("Menganalisis pola teks..."):
            time.sleep(0.4)
            pred_label, conf_score = predict_message(user_input, model)

        # Catatan: Sesuaikan pengecekan label di bawah ini dengan tipe target (y_train) di datasetmu.
        # Jika label di datasetmu berupa angka (1 dan 0), gunakan pred_label == 1.
        # Jika berupa teks ("spam" dan "ham"), bisa gunakan: if str(pred_label).lower() in ["1", "spam"]:
        if pred_label == 1 or str(pred_label).lower() == "spam":
            st.markdown(
                f"""
                <div class="result-card spam-card">
                    <div class="spam-title">Terdeteksi Sebagai SPAM</div>
                    <p class="spam-desc">
                        Pesan ini memiliki pola yang mencurigakan (seperti tawaran hadiah palsu, tautan berbahaya, atau promosi agresif). Harap berhati-hati dan jangan menekan tautan apa pun.
                    </p>
                    <div class="confidence-badge">Tingkat Keyakinan Model: {conf_score:.1f}%</div>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card ham-card">
                    <div class="ham-title">Pesan Aman (Bukan Spam)</div>
                    <p class="ham-desc">
                        Pesan ini terlihat seperti percakapan wajar dan tidak menunjukkan indikasi pola spam atau penipuan.
                    </p>
                    <div class="confidence-badge">Tingkat Keyakinan Model: {conf_score:.1f}%</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

# Menampilkan Footer
st.markdown(
    """
    <div class="footer">
        Powered by Machine Learning & Streamlit • Latihan LKSN-AI
    </div>
""",
    unsafe_allow_html=True,
)
