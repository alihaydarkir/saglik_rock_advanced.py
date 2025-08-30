# Sağlık ROCK Sistemi - Web Arayüzü (Streamlit)
# Ana sistemi import ederek web arayüzü sağlar

import streamlit as st
import time

# Ana sistem import - aynı klasörden
try:
    from saglik_rock_core import SaglikROCKSistemi
    CORE_SYSTEM_AVAILABLE = True
except ImportError:
    CORE_SYSTEM_AVAILABLE = False

# Streamlit sayfa config - en başta olmalı
st.set_page_config(
    page_title="🏥 Sağlık AI Asistanı", 
    page_icon="🏥",
    layout="wide"
)

# Global sistem - tek seferde başlatılır
@st.cache_resource
def saglik_sistemini_baslat():
    """Ana ROCK sistemini başlat"""
    if not CORE_SYSTEM_AVAILABLE:
        st.error("❌ saglik_rock_core.py bulunamadı! Aynı klasöre koyun.")
        st.stop()
        return None
        
    with st.spinner("🚀 Sağlık AI sistemi başlatılıyor..."):
        try:
            saglik_ai = SaglikROCKSistemi()
            basarili = saglik_ai.sistem_baslat()
            if basarili:
                st.success("✅ Sistem başarıyla yüklendi!")
                return saglik_ai
            else:
                st.error("❌ Sistem başlatılamadı!")
                return None
        except Exception as e:
            st.error(f"❌ Sistem hatası: {str(e)}")
            return None

def arama_yap(sorgu: str, sistem):
    """Ana ROCK sistemi ile arama yap"""
    if not sistem:
        st.error("❌ Sistem yüklenmedi! Sayfayı yenileyin.")
        return
    
    with st.spinner("🔍 Arama yapılıyor..."):
        try:
            # Ana ROCK sistemini kullan
            sonuc = sistem.soru_sor(sorgu)
        except Exception as e:
            st.error(f"❌ Arama hatası: {str(e)}")
            return
    
    # Sonuçları göster
    st.header("💡 Yanıt")
    
    # Acil durum uyarısı
    if sonuc.get('acil_durum', False):
        st.error("🚨 **ACIL DURUM!** Bu ciddi bir durum olabilir. Derhal 112'yi arayın!")
    
    if sonuc.get('basarili', False):
        st.success("✅ İlgili bilgi bulundu!")
        
        # Ana yanıt kutusu
        with st.container():
            st.markdown("### 📖 AI Analizi")
            st.markdown(sonuc.get('icerik', 'İçerik bulunamadı'))
            
            # Sistem detayları
            st.info(f"""
            📊 **Sistem Detayları**
            - Bulunan döküman: {sonuc.get('bulunan_dokuman_sayisi', 0)}
            - Sistem performansı: {sonuc.get('sistem_performansi', 'N/A')}
            - Güvenilirlik: Yüksek
            """)
            
            # Uyarı mesajı
            st.warning("""
            ⚠️ **ÖNEMLİ:** Bu bilgiler genel bilgilendirme amaçlıdır.
            - Kesin tanı için doktor kontrolü gerekir
            - İlaç kullanımı öncesi hekim onayı alın
            - Acil durumlarda 112'yi arayın
            """)
    else:
        st.error("❌ Maalesef ilgili güvenilir bilgi bulunamadı.")
        st.info("""
        💡 **Öneriler:**
        - Sorunuzu farklı kelimelerle ifade edin
        - Daha spesifik sorular sorun
        - Mutlaka bir sağlık uzmanı ile görüşün
        """)
    
    # Debug bilgileri
    if st.checkbox("🔧 Geliştirici Detayları"):
        st.subheader("Debug Bilgileri")
        st.json(sonuc)

def main():
    """Ana uygulama fonksiyonu"""
    
    # Başlık ve açıklama
    st.title("🏥 Sağlık AI Asistanı")
    st.markdown("### ROCK Metodolojisi ile Gelişmiş Tıbbi Bilgi Sistemi")
    
    # Sidebar - Sistem bilgileri
    with st.sidebar:
        st.header("📋 Sistem Bilgileri")
        st.info("""
        **🔬 ROCK Metodolojisi**
        - Retrieval: Semantik arama
        - Control: Güvenlik filtreleme  
        - Knowledge: Tıbbi veri bankası
        
        **🧠 AI Modeli**
        - Sentence Transformers
        - Multilingual support
        - Türkçe optimized
        """)
        
        st.header("⚠️ Önemli Uyarı")
        st.warning("""
        Bu sistem sadece **bilgilendirme** amaçlıdır.
        
        🚨 **Acil durumlar için 112'yi arayın**
        
        💊 **Kesin tanı ve tedavi için doktora başvurun**
        """)
    
    # Sistem başlat
    sistem = saglik_sistemini_baslat()
    
    # Ana içerik alanı
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Soru Sorun")
        
        # Soru girişi
        kullanici_sorusu = st.text_area(
            "Sağlık ile ilgili sorunuzu yazın:",
            placeholder="Örnek: Ateşim var ne yapmalıyım?",
            height=100
        )
        
        # Hızlı sorular
        st.subheader("⚡ Hızlı Sorular")
        hizli_sorular = [
            "Ateşim var ne yapmalıyım?",
            "Baş ağrım geçmiyor",
            "Şeker hastalığı nedir?",
            "Vitamin D eksikliği belirtileri",
            "Tansiyonum yüksek ne yapmalıyım?"
        ]
        
        selected_question = st.selectbox("Hazır sorulardan seçin:", [""] + hizli_sorular)
        if selected_question:
            kullanici_sorusu = selected_question
        
        # Arama butonu
        if st.button("🔍 Ara", type="primary", use_container_width=True):
            if kullanici_sorusu.strip():
                arama_yap(kullanici_sorusu.strip(), sistem)
            else:
                st.error("⚠️ Lütfen bir soru yazın!")
    
    with col2:
        st.header("📊 İstatistikler")
        
        # Sistem varsa istatistikleri göster
        if sistem:
            st.metric("Toplam Sorgu", sistem.sorgu_sayisi)
            st.metric("Başarılı Yanıt", sistem.basarili_sorgu)
            
            if sistem.sorgu_sayisi > 0:
                basari_orani = (sistem.basarili_sorgu / sistem.sorgu_sayisi) * 100
                st.metric("Başarı Oranı", f"{basari_orani:.1f}%")
        else:
            st.warning("Sistem henüz yüklenmedi")
            
        # Sistem durumu
        st.subheader("🔧 Sistem Durumu")
        if CORE_SYSTEM_AVAILABLE:
            st.success("✅ Core sistem mevcut")
        else:
            st.error("❌ Core sistem bulunamadı")
            
        if sistem:
            st.success("✅ AI sistemi aktif")
        else:
            st.error("❌ AI sistemi yüklenemedi")

# Ana uygulama çalıştır
if __name__ == "__main__":
    main()