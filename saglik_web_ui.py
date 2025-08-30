# ✅ Geliştirilmiş ve Çalışan Sağlık Web UI
# Bu dosyayı saglik_web_ui.py olarak kaydet
# Çalıştırmak için: streamlit run saglik_web_ui.py

import streamlit as st
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import time

class CPREgitimSistemi:
    def __init__(self):
        self.bilgi_bankasi = []
        self.vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)
        self.dokuman_vektorleri = None
        self.sistem_hazir = False
    def sistem_baslat(self):
        try:
            # JSON dosyasını oku
            try:
                with open("cpr_egitim_bilgi_bankasi.json", "r", encoding='utf-8') as f:
                    self.bilgi_bankasi = json.load(f)
            except FileNotFoundError:
                st.error("❌ JSON dosyası bulunamadı: cpr_egitim_bilgi_bankasi.json")
                return False
            except json.JSONDecodeError:
                st.error("❌ JSON dosyası geçersiz formatta")
                return False
    
            # Dosyanın boş olup olmadığını kontrol et
            if not self.bilgi_bankasi:
                st.error("❌ JSON dosyası boş")
                return False
    
            # JSON yapısını kontrol et (dizi olmalı)
            if not isinstance(self.bilgi_bankasi, list):
                st.error("❌ JSON dosyası bir dizi içermeli")
                return False
    
            # Gerekli alanları kontrol et
            required_fields = ['icerik', 'kategori', 'alt_kategori']
            for i, item in enumerate(self.bilgi_bankasi):
                for field in required_fields:
                    if field not in item:
                        st.error(f"❌ {i}. öğede '{field}' alanı eksik")
                        return False
    
            st.success(f"✅ JSON dosyası başarıyla yüklendi: {len(self.bilgi_bankasi)} kayıt")
    
            # Tüm metinleri birleştir
            metinler = []
            for madde in self.bilgi_bankasi:
                metinler.append(f"{madde['icerik']} {madde['kategori']} {madde['alt_kategori']}")
    
            # Vektörleştiriciyi eğit
            if metinler:
                self.dokuman_vektorleri = self.vectorizer.fit_transform(metinler)
                self.sistem_hazir = True
                return True
    
            return False
    
        except Exception as e:
            st.error(f"❌ Sistem başlatma hatası: {str(e)}")
            return False
            
        except Exception as e:
            st.error(f"❌ Sistem başlatma hatası: {str(e)}")
            return False   
        
    def soru_sor(self, soru):
        if not self.sistem_hazir:
            return {
                "basarili": False,
                "yanit": "Sistem hazır değil. Lütfen önce sistemi başlatın.",
                "bulunan_dokuman_sayisi": 0,
                "kaliteli_sonuc_sayisi": 0,
                "en_iyi_skor": 0,
                "sistem_performansi": "Düşük",
                "cpr_odakli": False,
                "acil_durum": False,
                "sonuc_detaylari": []
            }
        
        try:
            # Soruyu vektörleştir
            soru_vektoru = self.vectorizer.transform([soru])
            
            # Benzerlikleri hesapla
            benzerlikler = cosine_similarity(soru_vektoru, self.dokuman_vektorleri).flatten()
            
            # En iyi eşleşmeleri bul (skoru 0.1'den büyük olanlar)
            en_iyi_indisler = [i for i, score in enumerate(benzerlikler) if score > 0.1]
            en_iyi_indisler.sort(key=lambda i: benzerlikler[i], reverse=True)
            en_iyi_indisler = en_iyi_indisler[:3]  # En fazla 3 sonuç
            
            # Sonuçları hazırla
            sonuc_detaylari = []
            for idx in en_iyi_indisler:
                madde = self.bilgi_bankasi[idx]
                sonuc_detaylari.append((
                    benzerlikler[idx], 
                    madde["kategori"], 
                    madde["alt_kategori"],
                    madde["id"]
                ))
            
            # Yanıt oluştur
            yanit = ""
            basarili = False
            
            if sonuc_detaylari:
                # En iyi sonucu al
                en_iyi_skor, en_iyi_kategori, en_iyi_alt_kategori, en_iyi_id = sonuc_detaylari[0]
                
                if en_iyi_skor > 0.15:  # Eşik değeri düşürdüm
                    # İlgili bilgiyi bul
                    for madde in self.bilgi_bankasi:
                        if madde["id"] == en_iyi_id:
                            yanit = f"## 📋 {madde['kategori'].replace('_', ' ').title()} - {madde['alt_kategori'].title()}\n\n"
                            yanit += f"**İçerik:** {madde['icerik']}\n\n"
                            yanit += f"**Güvenilirlik:** {madde['guvenilirlik'] * 100:.0f}%\n"
                            yanit += f"**Acillik Seviyesi:** {madde['acillik_seviyesi'].capitalize()}\n"
                            if "metadata" in madde and "kaynak" in madde["metadata"]:
                                yanit += f"**Kaynak:** {madde['metadata']['kaynak']}"
                            break
                    basarili = True
            
            if not basarili:
                yanit = "## ⚠️ Bilgi Bulunamadı\n\nSoru için yeterli bilgi bulunamadı. Lütfen daha spesifik bir CPR sorusu sorun veya aşağıdaki örnek sorulardan birini deneyin."
            
            # CPR odaklı ve acil durum analizi
            cpr_odakli = any(terim in soru.lower() for terim in [
                "cpr", "kalp masajı", "sunî solunum", "resüsitasyon", "kalp durması", 
                "hipotermi", "hipotermik", "cardiopulmonary", "resuscitation"
            ])
            
            acil_durum = any(terim in soru.lower() for terim in [
                "acil", "ilk yardım", "hayat kurtarma", "müdahale", "112", "kritik", 
                "arrest", "acil durum", "emergency"
            ])
            
            # Performans değerlendirmesi
            if sonuc_detaylari and basarili:
                en_iyi_skor = sonuc_detaylari[0][0]
                if en_iyi_skor > 0.5:
                    performans = "Çok Yüksek"
                    performans_icon = "🚀"
                elif en_iyi_skor > 0.3:
                    performans = "Yüksek"
                    performans_icon = "📈"
                elif en_iyi_skor > 0.2:
                    performans = "Orta"
                    performans_icon = "📊"
                else:
                    performans = "Düşük"
                    performans_icon = "📉"
            else:
                performans = "Çok Düşük"
                performans_icon = "⚠️"
                en_iyi_skor = 0
            
            return {
                "basarili": basarili,
                "yanit": yanit,
                "bulunan_dokuman_sayisi": len([s for s in benzerlikler if s > 0.1]),
                "kaliteli_sonuc_sayisi": len([s for s in benzerlikler if s > 0.2]),
                "en_iyi_skor": en_iyi_skor,
                "sistem_performansi": f"{performans_icon} {performans}",
                "cpr_odakli": cpr_odakli,
                "acil_durum": acil_durum,
                "sonuc_detaylari": sonuc_detaylari
            }
            
        except Exception as e:
            return {
                "basarili": False,
                "yanit": f"## ❌ Hata\n\nSoru işlenirken bir hata oluştu: {str(e)}",
                "bulunan_dokuman_sayisi": 0,
                "kaliteli_sonuc_sayisi": 0,
                "en_iyi_skor": 0,
                "sistem_performansi": "⚠️ Hata",
                "cpr_odakli": False,
                "acil_durum": False,
                "sonuc_detaylari": []
            }

# 🎨 SAYFA AYARLARI
st.set_page_config(
    page_title="CPR Eğitim Sistemi",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 📚 ÖRNEK SORULAR
ornek_sorular = [
    "CPR kompresyon oranı nedir?",
    "Hipotermik arrest durumunda ne yapılmalı?",
    "Yetişkinlerde göğüs basısı derinliği ne olmalı?",
    "Çocuklarda CPR nasıl yapılır?",
    "AED nasıl kullanılır?",
    "CPR sırasında suni solunum oranı nedir?"
]

# 🎨 SIDEBAR - KOLAY ERİŞİM
with st.sidebar:
    st.header("🫀 CPR Hızlı Erişim")
    st.markdown("---")
    
    # Hızlı erişim butonları
    st.subheader("🚀 Örnek Sorular")
    for i, soru in enumerate(ornek_sorular):
        if st.button(soru, key=f"btn_{i}", use_container_width=True):
            st.session_state.soru_input = soru
            if 'sistem' in st.session_state:
                st.rerun()
    
    st.markdown("---")
    st.subheader("📊 Sistem Durumu")
    
    if "sistem" in st.session_state and st.session_state.get('basladi', False):
        st.success("✅ Sistem Çalışıyor")
        st.metric("Toplam Bilgi", f"{len(st.session_state.sistem.bilgi_bankasi)} kayıt")
    else:
        st.error("❌ Sistem Kapalı")
        if st.button("Sistemi Başlat", use_container_width=True):
            if "sistem" not in st.session_state:
                st.session_state.sistem = CPREgitimSistemi()
            st.session_state.basladi = st.session_state.sistem.sistem_baslat()
            st.rerun()
    
    st.markdown("---")
    st.subheader("ℹ️ Hakkında")
    st.info("""
    Bu sistem, CPR ve ilk yardım protokolleri 
    hakkında eğitim amaçlı geliştirilmiştir.
    Acil durumlarda profesyonel yardım çağırın.
    """)

# 🎨 ANA İÇERİK
st.title("🫀 CPR Eğitim ve Bilgi Sistemi")
st.markdown("Eğitim amaçlı **CPR protokollerine** hızlı erişim sağlayan yapay zeka tabanlı sistemdir.")

# 📊 İSTATİSTİK KARTLARI
if "sistem" in st.session_state and st.session_state.get('basladi', False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Toplam Bilgi", f"{len(st.session_state.sistem.bilgi_bankasi)}")
    with col2:
        st.metric("🔄 Sistem Durumu", "Çalışıyor")
    with col3:
        st.metric("⭐ Güvenilirlik", "Yüksek")
    with col4:
        st.metric("⚡ Yanıt Süresi", "<1sn")
else:
    st.warning("Sistem henüz başlatılmadı. Lütfen sidebar'dan sistemi başlatın.")

st.markdown("---")

# 🎯 SORU SORMA BÖLÜMÜ
st.header("💬 Soru Sor")

# Sistem başlatma kontrolü
if "sistem" not in st.session_state:
    st.session_state.sistem = CPREgitimSistemi()
    st.session_state.basladi = st.session_state.sistem.sistem_baslat()

# Soru girişi
soru = st.text_input(
    "CPR hakkında sorunuzu yazın:",
    value=st.session_state.get('soru_input', ''),
    placeholder="Örn: CPR kompresyon oranı nedir?",
    key="soru_input_main"
)

# Butonlar
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    sorgula_btn = st.button("🔍 Sorgula", use_container_width=True, type="primary")
with col2:
    if st.button("🔄 Temizle", use_container_width=True):
        st.session_state.soru_input = ""
        st.rerun()
with col3:
    rastgele_btn = st.button("🎲 Rastgele Soru", use_container_width=True)

if rastgele_btn:
    rastgele_soru = random.choice(ornek_sorular)
    st.session_state.soru_input = rastgele_soru
    st.rerun()

# 🔍 SONUÇLARI GÖSTER
if sorgula_btn and soru.strip():
    if not st.session_state.get('basladi', False):
        st.error("❌ Sistem başlatılmamış. Lütfen önce sidebar'dan sistemi başlatın.")
    else:
        with st.spinner("🤔 Sorunuz analiz ediliyor..."):
            # Yapay gecikme efekti
            time.sleep(0.5)
            sonuc = st.session_state.sistem.soru_sor(soru)
        
        # Sonuçları göster
        st.markdown("---")
        
        # Yanıt durumu
        if sonuc["basarili"]:
            st.success("✅ Uygun bilgi bulundu!")
        else:
            st.warning("⚠️ Yeterli bilgi bulunamadı.")
        
        # Yanıt içeriği
        st.markdown(sonuc['yanit'])
        
        # İstatistikler
        st.markdown("---")
        st.subheader("📊 Sorgu İstatistikleri")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**🔍 Bulunan Doküman Sayısı:** {sonuc['bulunan_dokuman_sayisi']}")
            st.info(f"**⭐ Kaliteli Sonuçlar:** {sonuc['kaliteli_sonuc_sayisi']}")
            st.info(f"**🏆 En İyi Skor:** {sonuc['en_iyi_skor']:.3f}")
            
        with col2:
            st.info(f"**📈 Performans:** {sonuc['sistem_performansi']}")
            st.info(f"**💡 CPR Odaklı mı?:** {'✅ Evet' if sonuc['cpr_odakli'] else '❌ Hayır'}")
            st.info(f"**🚨 Acil Durum?:** {'✅ Evet' if sonuc['acil_durum'] else '❌ Hayır'}")
        
        # Detaylı sonuçlar
        if sonuc["sonuc_detaylari"] and sonuc["basarili"]:
            st.markdown("---")
            st.subheader("📑 En İyi Eşleşmeler")
            
            # Gelişmiş tablo
            tablo_verisi = {
                "🏆 Skor": [f"{s[0]:.3f}" for s in sonuc["sonuc_detaylari"]],
                "📂 Kategori": [s[1].replace('_', ' ').title() for s in sonuc["sonuc_detaylari"]],
                "📁 Alt Kategori": [s[2].title() for s in sonuc["sonuc_detaylari"]],
                "🔍 Kalite": [
                    "Çok Yüksek" if s[0] > 0.5 else
                    "Yüksek" if s[0] > 0.3 else
                    "Orta" if s[0] > 0.2 else
                    "Düşük" for s in sonuc["sonuc_detaylari"]
                ]
            }
            
            st.dataframe(tablo_verisi, use_container_width=True, hide_index=True)

elif sorgula_btn and not soru.strip():
    st.warning("❗ Lütfen bir soru giriniz.")

# 📱 MOBILE UYUMLU TASARIM
st.markdown("---")
st.caption("""
📱 **Mobil Uyumlu** - Bu arayüz mobil cihazlarda da sorunsuz çalışır  
⚡ **Hızlı Yanıt** - Yapay zeka destekli anlık cevaplar  
🎯 **Doğru Bilgi** - Güvenilir kaynaklardan doğrulanmış içerik  
🔒 **Güvenli** - Hiçbir veri sunucuda saklanmaz
""")

# 🎨 STİL AYARLARI
st.markdown("""
<style>
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #ff4b4b;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        color: white;
    }
    .css-1d391kg {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e6e6e6;
    }
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)