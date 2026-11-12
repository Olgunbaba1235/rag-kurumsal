import streamlit as st
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

st.set_page_config(page_title="Enterprise RAG AI Asistanı", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1E3A8A;}
    .sub-header {font-size: 1.1rem; color: #4B5563;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Enterprise RAG Engine: Kurumsal Doküman & PDF AI Asistanı</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">TF-IDF Vektör Uzayı ve Cosine Similarity (Kosinüs Benzerliği) algoritması ile şirket dokümanlarında anlamsal arama yapan yapay zeka asistanı.</div>', unsafe_allow_html=True)
st.divider()

default_text = """MADDE 1 - ÇALIŞMA SAATLERİ VE HİBRİT MODEL:
Şirketimizde mesai saatleri hafta içi 09:00 - 18:00 arasındadır. Yazılım, Veri Bilimi ve Yapay Zeka ekipleri haftada 3 gün uzaktan (evden), 2 gün ofisten olacak şekilde hibrit çalışma hakkına sahiptir. Cuma günleri tüm şirket uzaktan çalışmaktadır.

MADDE 2 - YILLIK İZİN VE MAZERET İZİNLERİ:
Tam zamanlı personelin yıllık ücretli izin hakkı ilk yıl için 14 iş günüdür. Şirkette 5 yılını dolduran kıdemli uzmanlarda bu süre 20 iş gününe çıkar. Doğum günü olan çalışanlar o gün idari izinli sayılır.

MADDE 3 - BİLGİ GÜVENLİĞİ, KVKK VE YAPAY ZEKA PROTOKOLÜ:
Şirket içi müşteri verileri ve kaynak kodları açık kaynaklı dış sistemlere (genel LLM'lere) yüklenemez. Harici USB bellek kullanımı tamamen yasaktır. Tüm veri analitiği işlemleri şirketin kendi izole bulut sunucuları (VPC) üzerinden yürütülmelidir.

MADDE 4 - SEYAHAT, KONAKLAMA VE HARCIRAH POLİTİKASI:
Şirket dışı müşteri ziyaretlerinde ve teknoloji konferanslarında konaklama bütçesi gecelik maksimum 5.000 TL'dir. Günlük yemek harcırahı 1.200 TL olarak belirlenmiştir. Fiş ve faturalar her ayın 5'ine kadar ERP sistemine yüklenmelidir.

MADDE 5 - EĞİTİM, SERTİFİKA VE DONANIM DESTEĞİ:
Çalışanların mesleki gelişimi için alınan AWS, Google Cloud, Veri Bilimi ve Yabancı Dil (İngilizce) eğitimlerinin ve sınav ücretlerinin %100'ü şirket bütçesinden karşılanır. Ayrıca her yazılımcıya 2 yılda bir donanım yenileme bütçesi verilir.

MADDE 6 - PERFORMANS PRİMİ VE HİSSE (STOCK OPTION) HAKLARI:
Yıl sonu performans değerlendirmesinde 'Üstün Başarı' gösteren çalışanlara 3 maaş tutarında nakit prim ödenir. 2 yılını dolduran mühendisler şirketin hisse edindirme (Stock Option) programına dahil edilir."""

if os.path.exists("kurumsal_kilavuz.txt"):
    with open("kurumsal_kilavuz.txt", "r", encoding="utf-8") as f:
        default_text = f.read()

with st.sidebar:
    st.header("⚙️ RAG Motoru Ayarları")
    st.markdown("**Arama Algoritması:** `TF-IDF + Cosine Similarity`")
    esik_degeri = st.slider("Minimum Güven Skoru Eşiği (%)", min_value=5, max_value=50, value=12, step=1)
    top_k = st.selectbox("Getirilecek Maksimum Kaynak (Top-K)", [1, 2, 3], index=1)
    st.divider()
    st.caption("🚀 Geliştirici: Olgun Dağdelen | AI Portfolio")

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("📂 Kurumsal Bilgi Tabanı (Knowledge Base)")
    uploaded_file = st.file_uploader("Şirket Yönetmeliği / Sözleşme Yükleyin (.pdf veya .txt)", type=["pdf", "txt"])
    
    active_text = default_text
    kaynak_adi = "Şirket Anayasası ve Prosedürler (Varsayılan)"
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            pdf_text = [page.extract_text() for page in reader.pages if page.extract_text()]
            active_text = "\n\n".join(pdf_text)
            kaynak_adi = uploaded_file.name
            st.success(f"✅ PDF Başarıyla İndekslendi: {uploaded_file.name} ({len(reader.pages)} Sayfa)")
        else:
            active_text = uploaded_file.read().decode("utf-8")
            kaynak_adi = uploaded_file.name
            st.success(f"✅ Metin Belgesi Başarıyla İndekslendi: {uploaded_file.name}")
    else:
        st.info("📌 Şu an varsayılan '2026 Kurumsal Şirket Prosedürleri' veritabanı aktif.")

    chunks = [p.strip() for p in active_text.split("\n\n") if len(p.strip()) > 20]
    
    c1, c2 = st.columns(2)
    c1.metric("İndekslenen Paragraf (Chunk)", len(chunks))
    c2.metric("Toplam Kelime Sayısı", len(active_text.split()))
    
    with st.expander("📄 Aktif Doküman İçeriğini İncele", expanded=False):
        st.text_area("Ham Metin:", active_text, height=250)

with col2:
    st.subheader("💬 Kurumsal AI Asistan ile Konuşun")
    st.markdown("**⚡ Hızlı Test Soruları:**")
    b1, b2, b3 = st.columns(3)
    hizli_soru = None
    if b1.button("🏠 Uzaktan çalışma kaç gün?"):
        hizli_soru = "Uzaktan çalışma ve mesai saatleri nasıl uygulanıyor?"
    if b2.button("🎓 Eğitim ve İngilizce desteği?"):
        hizli_soru = "Yabancı dil İngilizce ve bulut eğitim desteği bütçesi ne kadar?"
    if b3.button("💰 Prim ve Hisse hakları?"):
        hizli_soru = "Yıl sonu performans primi ve hisse senedi hakları nelerdir?"

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Merhaba! Ben Kurumsal RAG Asistanınız. Şu an **{kaynak_adi}** dokümanını hafızama aldım. Bana şirket kuralları, izinler, bütçeler veya yüklediğiniz PDF hakkında her şeyi sorabilirsiniz."}
        ]

    chat_container = st.container(height=320)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Dokümana sorunuzu yazın... (Örn: Konaklama bütçesi ne kadar?)")
    aktif_sorgu = hizli_soru if hizli_soru else user_input

    if aktif_sorgu:
        st.session_state.messages.append({"role": "user", "content": aktif_sorgu})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(aktif_sorgu)

        if len(chunks) > 0:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            all_texts = chunks + [aktif_sorgu]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            query_vec = tfidf_matrix[-1]
            chunk_vecs = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(query_vec, chunk_vecs).flatten()
            sorted_indices = np.argsort(similarities)[::-1]
            
            best_idx = sorted_indices[0]
            best_score = similarities[best_idx] * 100
            
            if best_score >= esik_degeri:
                yanit = f"🎯 **En Yüksek Eşleşen Kaynak (Güven Skoru: %{best_score:.1f}):**\n\n> {chunks[best_idx]}"
                ek_maddeler = []
                for idx in sorted_indices[1:top_k]:
                    skor = similarities[idx] * 100
                    if skor >= (esik_degeri / 2):
                        ek_maddeler.append(f"- *(Skor: %{skor:.1f})* {chunks[idx][:180]}...")
                if ek_maddeler:
                    yanit += "\n\n**📚 Diğer İlgili Referanslar:**\n" + "\n".join(ek_maddeler)
            else:
                yanit = f"⚠️ Doküman içerisinde bu soruyla yüksek oranda eşleşen bir madde bulunamadı *(En yakın eşleşme skoru: %{best_score:.1f}, Eşik: %{esik_degeri})*. Lütfen farklı anahtar kelimelerle tekrar sorunuz."
        else:
            yanit = "❌ Doküman boş veya okunamadı."

        st.session_state.messages.append({"role": "assistant", "content": yanit})
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(yanit)
