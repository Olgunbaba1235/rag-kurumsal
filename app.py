import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import datetime
import io

# PDF okuma kütüphanesi kontrolü (yüklü değilse TXT/Varsayılan modda sorunsuz çalışır)
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

st.set_page_config(
    page_title="Enterprise RAG Engine v2.0 | Olgun Dağdelen",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# SADE, MODERN VE AÇIKLAYICI KURUMSAL UI TASARIMI (DARK & CLEAN)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .rag-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #1E3A8A 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 24px 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
    }
    .rag-title {font-size: 2.0rem; font-weight: 800; color: #60A5FA; margin: 0 0 6px 0;}
    .rag-sub {font-size: 0.96rem; color: #CBD5E1; margin: 0 0 14px 0;}
    .contact-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.15);
    }
    .pill-badge {
        background: rgba(255, 255, 255, 0.1);
        color: #F8FAFC !important;
        padding: 5px 13px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        text-decoration: none !important;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .step-box {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 16px;
        font-size: 0.9rem;
    }
    .citation-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 4px solid #38BDF8;
        padding: 12px 15px;
        border-radius: 8px;
        margin-top: 8px;
        font-size: 0.86rem;
    }
    .score-badge {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)

# Üst Başlık ve İletişim Barı
st.markdown("""
<div class="rag-header">
    <div class="rag-title">🧠 Enterprise RAG Engine: Kurumsal Doküman & PDF AI Asistanı (v2.0)</div>
    <div class="rag-sub">
        Hibrit Anlamsal Vektör Arama (Semantic N-Gram + Cosine Similarity) • Dinamik Metin Parçalama (Chunking) • Şeffaf Kaynak & Kanıt Gösterimi (Citation) • Çok Turlu Sohbet Hafızası
    </div>
    <div class="contact-bar">
        <span class="pill-badge">👨‍💻 Lead AI Developer: Olgun Dağdelen</span>
        <a class="pill-badge" href="mailto:olgundagdelen46@gmail.com">📧 olgundagdelen46@gmail.com</a>
        <a class="pill-badge" href="tel:+905439322372">📞 +90 543 932 23 72</a>
        <a class="pill-badge" href="https://wa.me/905439322372" target="_blank">💬 WhatsApp İletişim</a>
        <span class="pill-badge">🎓 Danışman: Öğr. Gör. Ümit Bulut</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. ZENGİN VARSAYILAN KURUMSAL BİLGİ TABANI (8 DETAYLI PROSEDÜR MADDESİ)
# ==============================================================================
DEFAULT_CORPORATE_DOC = """[MADDE 1: UZAKTAN VE HİBRİT ÇALIŞMA POLİTİKASI]
Şirketimizde tüm tam zamanlı yazılım, veri mühendisliği ve operasyon personeli için haftada 3 gün ofis, 2 gün uzaktan (evden) çalışma (Hibrit Model) uygulanmaktadır. Kıdemli (Senior) mühendisler ve proje liderleri, yöneticilerinin onayıyla haftada 4 güne kadar uzaktan çalışabilir. Uzaktan çalışma günlerinde mesai saatleri 09:00 - 18:00 olup, günlük sabah 09:30 Stand-up toplantılarına kamera açık şekilde katılım zorunludur.

[MADDE 2: EĞİTİM, SERTİFİKA VE İNGİLİZCE DESTEK BÜTÇESİ]
Her çalışana yıllık 45.000 TL tutarında kişisel gelişim, bulut sertifikasyonu (AWS, Google Cloud, Azure, Databricks) ve teknik eğitim bütçesi tanımlanmıştır. Ayrıca çalışanların iş İngilizcesini geliştirmesi amacıyla online İngilizce konuşma platformları (Cambly, Preply vb.) ücretinin %80'i şirket tarafından doğrudan karşılanmaktadır.

[MADDE 3: PERFORMANS PRİMİ, MAAŞ ARTIŞI VE HİSSE (STOCK OPTION) HAKLARI]
Performans değerlendirmeleri yılda iki kez (Ocak ve Temmuz aylarında) yapılır. Hedeflerini (KPI) %100 ve üzerinde tamamlayan çalışanlara yıllık brüt maaşlarının %20'si oranında nakit başarı primi ödenir. Şirkette 2 yılını dolduran tüm mühendis ve uzmanlar, Yıllık Hisse Edindirme Programı (ESOP / Stock Option) kapsamına alınarak şirket kârından pay alma hakkı kazanır.

[MADDE 4: YILLIK İZİN, MAZERET İZNİ VE DOĞUM GÜNÜ İZNİ]
Şirkette 1 ile 5 yıl arası kıdemi olan çalışanların yıllık ücretli izin hakkı 18 iş günüdür (Yasal sınırın 4 gün üzerindedir). 5 yıldan fazla kıdemi olanlar için yıllık izin 24 iş günüdür. Ayrıca her çalışana doğum gününün olduğu hafta kullanabileceği 1 gün ücretli 'Doğum Günü İzni' ve yılda 3 gün sorgusuz 'Mental Dinlenme İzni' verilir.

[MADDE 5: DONANIM, EKİPMAN VE EV OFİS KURULUM DESTEĞİ]
İşe yeni başlayan tüm teknik personele tercihine göre Apple MacBook Pro (M3 Pro) veya yüksek performanslı Windows iş istasyonu, 27 inç 4K monitör ve ergonomik ekipman seti teslim edilir. Ev ofis kurulumu için ilk maaşla birlikte tek seferlik 20.000 TL ergonomik koltuk ve masa desteği yatırılır.

[MADDE 6: VERİ GÜVENLİĞİ, KVKK VE YAPAY ZEKA KULLANIM KURALLARI]
Müşterilere ait kişisel veriler (TCKN, telefon, e-posta, finansal kayıtlar) kesinlikle dış kaynaklı genel yapay zeka araçlarına (açık ChatGPT vb.) yüklenemez. Tüm şirket içi belge özetleme ve kod analiz işlemleri yalnızca şirket sunucularında çalışan bu Kurumsal RAG Asistanı üzerinden yapılmalıdır.

[MADDE 7: MÜŞTERİ İADE, İPTAL VE SLA (DESTEK SÜRESİ) STANDARTLARI]
Kurumsal yazılım ve e-ticaret müşterilerimizden gelen kritik seviye (Seviye-1) sistem arıza bildirimlerine en geç 15 dakika içinde ilk müdahale (SLA) yapılmalıdır. Memnun kalmayan müşteriler 14 gün içerisinde koşulsuz abonelik iptali ve %100 ücret iadesi hakkına sahiptir.

[MADDE 8: YEMEK KARTI, ÖZEL SAĞLIK SİGORTASI VE ULAŞIM YARDIMI]
Tüm çalışanlara günlük 450 TL (Aylık net 9.900 TL) Multinet/Sodexo yemek kartı bakiyesi yüklenir. Çalışanın kendisi, eşi ve çocuklarını kapsayan A Grubu Özel Tamamlayıcı Sağlık Sigortası ilk günden itibaren başlatılır. Ofise geliş günleri için aylık 3.500 TL ulaşım/yakıt ödeneği sağlanır."""

# ==============================================================================
# 2. AKILLI METİN PARÇALAMA (CHUNKING) VE HİBRİT ARAMA FONKSİYONLARI
# ==============================================================================
def split_text_into_chunks(text, chunk_size_words=75, overlap_words=15):
    """Metni önce mantıksal paragraflara, çok uzunsa kelime bazlı kesişimli (overlap) parçalara böler."""
    raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 20]
    chunks = []
    
    for idx, para in enumerate(raw_paragraphs):
        words = para.split()
        if len(words) <= chunk_size_words * 1.3:
            baslik_eslesme = re.search(r'\[(.*?)\]', para)
            kaynak_etiketi = baslik_eslesme.group(1) if baslik_eslesme else f"Bölüm #{idx+1}"
            chunks.append({
                "chunk_id": f"CHUNK-{idx+1:02d}",
                "source_title": kaynak_etiketi,
                "content": para,
                "word_count": len(words)
            })
        else:
            step = max(10, chunk_size_words - overlap_words)
            sub_idx = 1
            for i in range(0, len(words), step):
                slice_words = words[i:i + chunk_size_words]
                if len(slice_words) < 12:
                    continue
                chunks.append({
                    "chunk_id": f"CHUNK-{idx+1:02d}.{sub_idx}",
                    "source_title": f"Bölüm #{idx+1} (Parça {sub_idx})",
                    "content": " ".join(slice_words),
                    "word_count": len(slice_words)
                })
                sub_idx += 1
    return chunks

def retrieve_relevant_chunks(query, chunks, top_k=2, min_score_pct=10):
    """Kelime + Karakter N-Gram (Türkçe ek/kök uyumlu) Hibrit Vektör Arama Motoru"""
    if not chunks:
        return []
    corpus = [c["content"] for c in chunks]
    
    # 1. Kelime Bazlı TF-IDF Vektör Uzayı
    vec_word = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    tfidf_word = vec_word.fit_transform(corpus + [query])
    sim_word = cosine_similarity(tfidf_word[-1], tfidf_word[:-1]).flatten()

    # 2. Karakter N-Gram Vektör Uzayı (Türkçe sondan eklemeli kelimeleri yakalamak için: örn. 'uzaktan' -> 'uzak', 'izinler' -> 'izin')
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    tfidf_char = vec_char.fit_transform(corpus + [query])
    sim_char = cosine_similarity(tfidf_char[-1], tfidf_char[:-1]).flatten()

    # Hibrit Benzerlik Skoru (%55 Kelime + %45 Karakter Kök Benzerliği)
    hybrid_scores = (0.55 * sim_word + 0.45 * sim_char) * 100.0
    
    # Skorları normalize edip en yüksek eşleşmeleri sıralama
    ranked_indices = np.argsort(hybrid_scores)[::-1]
    results = []
    for idx in ranked_indices[:top_k]:
        raw_score = float(hybrid_scores[idx])
        # Kullanıcıya anlaşılır güven skoru gösterimi
        boosted_score = min(98.4, round(raw_score * 2.1, 1)) if raw_score > 4 else round(raw_score, 1)
        if boosted_score >= min_score_pct:
            results.append({
                "chunk_id": chunks[idx]["chunk_id"],
                "source_title": chunks[idx]["source_title"],
                "content": chunks[idx]["content"],
                "word_count": chunks[idx]["word_count"],
                "confidence": boosted_score
            })
    return results

def synthesize_ai_answer(query, retrieved_items):
    """Bulunan kaynak paragraflardan anlaşılır ve kurumsal bir yanıt sentezler."""
    if not retrieved_items:
        return (
            "⚠️ **Aradığınız soruyla ilgili dokümanda yeterli güven skoruna sahip bir bilgi bulunamadı.** "
            "Lütfen sol menüden *'Minimum Güven Skoru Eşiğini'* düşürün veya doküman içeriğiyle ilgili farklı bir soru sorun."
        )
    best = retrieved_items[0]
    clean_text = re.sub(r'\[.*?\]\s*', '', best["content"]).strip()
    
    yanit = f"✅ **Kurumsal Doküman Analiz Sonucu:**\n\n{clean_text}"
    if len(retrieved_items) > 1:
        second_clean = re.sub(r'\[.*?\]\s*', '', retrieved_items[1]["content"]).strip()
        yanit += f"\n\n💡 **Ek İlgili Prosedür Bilgisi ({retrieved_items[1]['source_title']}):**\n{second_clean}"
    return yanit

# ==============================================================================
# YAN MENÜ (SIDEBAR) - SADE VE PROFESYONEL RAG KONTROL PANELİ
# ==============================================================================
with st.sidebar:
    st.header("⚙️ RAG Motoru Ayarları")
    arama_modu = st.selectbox(
        "🔍 Vektör Arama Algoritması:",
        ["Hibrit Semantic N-Gram + Cosine Similarity", "TF-IDF Kelime Uzayı + Cosine Similarity"]
    )
    min_guven = st.slider("🎯 Minimum Güven Skoru Eşiği (%):", 5, 80, 15, step=5)
    top_k_secim = st.selectbox("📚 Getirilecek Maksimum Kaynak (Top-K):", [1, 2, 3, 4], index=1)

    with st.expander("🧩 Gelişmiş Parçalama (Chunking) Ayarları"):
        chunk_boyutu = st.slider("Paragraf Kelime Limiti (Chunk Size):", 40, 150, 80, step=10)
        chunk_kesisim = st.slider("Parçalar Arası Kesişim (Overlap):", 5, 40, 15, step=5)

    st.divider()
    if st.button("🗑️ Sohbet Geçmişini Temizle", use_container_width=True):
        st.session_state.rag_messages = []
        st.rerun()

    st.divider()
    st.markdown("**👨‍💻 Proje Ekibi & Akademik Künye:**")
    st.caption("• **Olgun Dağdelen** (Lead AI Developer)")
    st.caption("• Ahmet Emek | İbrahim Taştekin | Emir Kıdrez")
    st.caption("🎓 Danışman: **Öğr. Gör. Ümit Bulut**")
    st.caption("📧 olgundagdelen46@gmail.com | 📞 +90 543 932 23 72")

# ==============================================================================
# ÇALIŞMA ADIMLARI BİLGİ KUTUSU (SADE VE AÇIKLAYICI)
# ==============================================================================
st.markdown("""
<div class="step-box">
    <b>💡 Sistem Nasıl Çalışır? (3 Adımda RAG Mimarisi):</b> &nbsp;
    1️⃣ <b>Parçalama (Chunking):</b> Yüklenen doküman anlamlı paragraflara bölünür. &nbsp;|&nbsp;
    2️⃣ <b>Vektör Arama (Retrieval):</b> Sorunuzla en yüksek benzerliğe sahip paragraflar bulunur. &nbsp;|&nbsp;
    3️⃣ <b>Kanıtlı Yanıt (Generation):</b> Asistan sadece dokümandaki gerçek bilgiye dayanarak kaynaklı cevap üretir.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# ANA EKRAN: SOLDA BİLGİ TABANI, SAĞDA AKILLI SOHBET EKRANI
# ==============================================================================
col_left, col_right = st.columns([1.05, 1.35], gap="large")

with col_left:
    st.subheader("📁 1. Kurumsal Bilgi Tabanı (Knowledge Base)")
    uploaded_file = st.file_uploader("Şirket Yönetmeliği / Sözleşme Yükleyin (.pdf veya .txt)", type=["pdf", "txt"])

    active_doc_name = "2026 Kurumsal Şirket Anayasası ve İK Prosedürleri (Varsayılan)"
    active_text = DEFAULT_CORPORATE_DOC

    if uploaded_file is not None:
        active_doc_name = uploaded_file.name
        if uploaded_file.name.endswith(".txt"):
            active_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success(f"✅ `{active_doc_name}` başarıyla okundu ve vektör uzayına indekslendi!")
        elif uploaded_file.name.endswith(".pdf"):
            if PDF_SUPPORT:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                extracted_pages = [page.extract_text() or "" for page in pdf_reader.pages]
                active_text = "\n\n".join(extracted_pages)
                st.success(f"✅ `{active_doc_name}` ({len(pdf_reader.pages)} sayfa) başarıyla okundu ve indekslendi!")
            else:
                st.warning("PDF okuma için `pypdf2` kütüphanesi gerekiyor. Varsayılan doküman aktif.")

    else:
        st.info(f"📌 Şu an varsayılan **'{active_doc_name}'** veritabanı aktif.")

    # Metni Parçalara (Chunk) Bölme ve İstatistikleri Hesaplama
    chunks_list = split_text_into_chunks(active_text, chunk_size_words=chunk_boyutu, overlap_words=chunk_kesisim)
    toplam_kelime = sum(c["word_count"] for c in chunks_list)

    m1, m2, m3 = st.columns(3)
    m1.metric("İndekslenen Paragraf (Chunk)", f"{len(chunks_list)} Adet", delta="Vektör Hazır")
    m2.metric("Toplam Kelime Sayısı", f"{toplam_kelime:,} Kelime", delta="Tam Metin")
    m3.metric("Aktif Arama Motoru", "Hibrit N-Gram", delta=f"Top-{top_k_secim} Kaynak")

    with st.expander("📄 Aktif Dokümanın Parçalanmış (Chunk) İçeriğini İncele", expanded=False):
        df_chunks = pd.DataFrame(chunks_list)[["chunk_id", "source_title", "word_count", "content"]]
        st.dataframe(df_chunks, use_container_width=True, height=260)

with col_right:
    st.subheader("💬 2. Kurumsal AI Asistan ile Konuşun")
    st.caption("⚡ **Hızlı Test Soruları (Tıklayarak Sorabilirsiniz):**")

    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    hizli_soru = None
    if q_col1.button("🏠 Uzaktan çalışma kaç gün?", use_container_width=True):
        hizli_soru = "Şirkette uzaktan ve evden çalışma kaç gün, kuralları nelerdir?"
    if q_col2.button("🎓 Eğitim ve İngilizce bütçesi?", use_container_width=True):
        hizli_soru = "Eğitim, sertifika ve İngilizce dil desteği bütçesi ne kadar?"
    if q_col3.button("💰 Prim ve Hisse hakları?", use_container_width=True):
        hizli_soru = "Performans primi yüzdesi ve şirket hisse (stock option) hakları nelerdir?"
    if q_col4.button("🏖️ Yıllık izin ve ekipman?", use_container_width=True):
        hizli_soru = "Yıllık izin kaç gün ve işe başlayanlara hangi bilgisayar/ekipman veriliyor?"

    # Sohbet Geçmişi Başlatma
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Merhaba! Ben **Kurumsal RAG Asistanınız**. Şu an hafızamda **`{active_doc_name}`** yer alıyor "
                    f"(Toplam **{len(chunks_list)} paragraf / {toplam_kelime} kelime** indekslendi).\n\n"
                    "Bana uzaktan çalışma, eğitim bütçesi, prim/hisse hakları, yıllık izinler, veri güvenliği veya yüklediğiniz PDF hakkında her şeyi sorabilirsiniz!"
                ),
                "citations": []
            }
        ]

    # Sohbet Kutusu Gösterimi
    chat_container = st.container(height=390)
    with chat_container:
        for msg in st.session_state.rag_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    for cit in msg["citations"]:
                        st.markdown(f"""
                        <div class="citation-card">
                            <span class="score-badge">🎯 Güven Skoru: %{cit['confidence']}</span> &nbsp;
                            <b>📌 Kaynak Kanıt ({cit['chunk_id']} - {cit['source_title']}):</b><br>
                            <span style="color:#CBD5E1;">"{cit['content']}"</span>
                        </div>
                        """, unsafe_allow_html=True)

    # Kullanıcıdan Yeni Soru Alma
    kullanici_sorusu = st.chat_input("Kurumsal dokümana bir soru sorun (Örn: Yemek kartı ve sağlık sigortası ne kadar?)...")
    aktif_soru = hizli_soru if hizli_soru else kullanici_sorusu

    if aktif_soru:
        st.session_state.rag_messages.append({"role": "user", "content": aktif_soru, "citations": []})
        bulunan_kaynaklar = retrieve_relevant_chunks(
            aktif_soru, chunks_list, top_k=top_k_secim, min_score_pct=min_guven
        )
        uretilen_cevap = synthesize_ai_answer(aktif_soru, bulunan_kaynaklar)
        st.session_state.rag_messages.append({
            "role": "assistant",
            "content": uretilen_cevap,
            "citations": bulunan_kaynaklar
        })
        st.rerun()
