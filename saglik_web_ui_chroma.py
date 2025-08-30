# ✅ Geliştirilmiş ve Birleştirilmiş Sağlık Web UI
# Bu dosyayı saglik_web_ui_chroma.py olarak kaydet
# Çalıştırmak için: streamlit run saglik_web_ui_chroma.py

import streamlit as st
import json
import numpy as np
import random
import time
import uuid
from typing import List, Dict

# ChromaDB ve Sentence Transformers için try-catch
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    st.warning("❌ ChromaDB kurulu değil. Kurulum: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("❌ Sentence Transformers kurulu değil. Kurulum: pip install sentence-transformers")

class ChromaRetriever:
    """ChromaDB ile hızlı vector arama sistemi"""
    
    def __init__(self, koleksiyon_adi: str = "cpr_egitim_bilgi_bankasi"):
        self.koleksiyon_adi = koleksiyon_adi
        self.chroma_client = None
        self.koleksiyon = None
        self.model = None
        
    def sistem_baslat(self):
        """ChromaDB ve model başlatma"""
        if not CHROMA_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            st.error("❌ Gerekli kütüphaneler eksik! pip install chromadb sentence-transformers")
            return False
        
        try:
            # ChromaDB client oluştur
            self.chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            
            # Model yükle
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Koleksiyon oluştur veya getir
            try:
                self.koleksiyon = self.chroma_client.get_collection(self.koleksiyon_adi)
                st.success(f"✅ Mevcut koleksiyon yüklendi: {self.koleksiyon.count()} doküman")
            except:
                self.koleksiyon = self.chroma_client.create_collection(
                    name=self.koleksiyon_adi,
                    metadata={"description": "CPR eğitim vektör veritabanı"}
                )
                st.success("✅ Yeni koleksiyon oluşturuldu")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Sistem başlatma hatası: {str(e)}")
            return False
    
    def dokumanlar_ekle(self, dokumanlar: List[Dict], temizle: bool = False):
        """Dökümanları vector database'e ekle"""
        if not self.koleksiyon or not self.model:
            st.error("❌ Sistem başlatılmamış!")
            return False
        
        if temizle:
            # Mevcut koleksiyonu temizle
            try:
                self.chroma_client.delete_collection(self.koleksiyon_adi)
            except:
                pass
            self.koleksiyon = self.chroma_client.create_collection(
                name=self.koleksiyon_adi,
                metadata={"description": "CPR eğitim vektör veritabanı"}
            )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Batch işleme için veriler
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for i, dok in enumerate(dokumanlar):
            # Benzersiz ID oluştur
            doc_id = dok.get('id', str(uuid.uuid4()))
            ids.append(doc_id)
            
            # İçerik embedding'i oluştur
            icerik = dok['icerik']
            embedding = self.model.encode(icerik).tolist()
            embeddings.append(embedding)
            
            # Metadata hazırla
            metadata = {
                'kategori': dok.get('kategori', 'genel'),
                'alt_kategori': dok.get('alt_kategori', 'genel'),
                'guvenilirlik': float(dok.get('guvenilirlik', 0.8)),
                'acillik_seviyesi': dok.get('acillik_seviyesi', 'normal'),
                'kaynak': dok.get('metadata', {}).get('kaynak', 'bilinmiyor')
            }
            metadatas.append(metadata)
            
            # Tam döküman metni
            documents.append(icerik)
            
            # İlerleme güncelleme
            if len(dokumanlar) > 10 and (i + 1) % max(1, len(dokumanlar) // 10) == 0:
                progress = (i + 1) / len(dokumanlar)
                progress_bar.progress(progress)
                status_text.text(f"📊 Dokümanlar işleniyor: {i + 1}/{len(dokumanlar)}")
        
        # Batch olarak ekle
        try:
            self.koleksiyon.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
            progress_bar.progress(1.0)
            status_text.text(f"✅ {len(dokumanlar)} doküman vector database'e eklendi!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            return True
            
        except Exception as e:
            st.error(f"❌ Doküman ekleme hatası: {str(e)}")
            return False
    
    def hizli_arama(self, sorgu: str, top_k: int = 10) -> List[Dict]:  # top_k 10'a çıkarıldı
        if not self.koleksiyon or not self.model:
            st.error("❌ Sistem başlatılmamış!")
            return []
        
        try:
            # Sorgu embedding'i oluştur
            sorgu_embedding = self.model.encode(sorgu).tolist()
            
            # ChromaDB'den ara - DAHA FAZLA SONUÇ
            sonuclar = self.koleksiyon.query(
                query_embeddings=[sorgu_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Sonuçları formatla
            arama_sonuclari = []
            if sonuclar['documents'] and len(sonuclar['documents'][0]) > 0:
                for i in range(len(sonuclar['documents'][0])):
                    # Cosine benzerliğini hesapla (distance -> similarity)
                    distance = sonuclar['distances'][0][i]
                    similarity = max(0.1, 1 - distance)  # Minimum 0.1 similarity
                    
                    sonuc = {
                        'id': sonuclar['ids'][0][i],
                        'icerik': sonuclar['documents'][0][i],
                        'benzerlik_skoru': similarity,
                        'metadata': sonuclar['metadatas'][0][i],
                        'kategori': sonuclar['metadatas'][0][i].get('kategori', 'bilinmiyor'),
                        'guvenilirlik': sonuclar['metadatas'][0][i].get('guvenilirlik', 0.8)
                    }
                    arama_sonuclari.append(sonuc)
            
            # Benzerlik skoruna göre sırala
            arama_sonuclari.sort(key=lambda x: x['benzerlik_skoru'], reverse=True)
            
            # DEBUG: Sonuçları göster
            print(f"[DEBUG] '{sorgu}' için {len(arama_sonuclari)} sonuç:")
            for i, sonuc in enumerate(arama_sonuclari[:3]):
                print(f"  {i+1}. Skor: {sonuc['benzerlik_skoru']:.3f}, Kategori: {sonuc['kategori']}")
            
            return arama_sonuclari
            
        except Exception as e:
            st.error(f"❌ Arama hatası: {str(e)}")
            return []
    
    def soru_sor(self, soru):
        self.sorgu_sayisi += 1
        
        # ChromaDB ile arama yap - DAHA ÇOK SONUÇ
        sonuclar = self.retriever.hizli_arama(soru, top_k=8)
        
        # CPR odaklı ve acil durum analizi
        cpr_odakli = any(terim in soru.lower() for terim in [
            "cpr", "kalp masajı", "sunî solunum", "resüsitasyon", "kalp durması", 
            "hipotermi", "hipotermik", "cardiopulmonary", "resuscitation",
            "kompresyon", "aed", "defibrilatör", "canlandırma", "ilk yardım"
        ])
        
        acil_durum = any(terim in soru.lower() for terim in [
            "acil", "ilk yardım", "hayat kurtarma", "müdahale", "112", "kritik", 
            "arrest", "acil durum", "emergency", "kriz", "müdahale"
        ])
        
        # ✅ BENZERLİK EŞİĞİNİ DÜŞÜRDÜM (0.3 -> 0.2)
        kaliteli_sonuclar = [s for s in sonuclar if s['benzerlik_skoru'] > 0.2]
        
        # ✅ EN AZ 1 SONUÇ VARSA KABUL ET
        if len(kaliteli_sonuclar) >= 1:
            self.basarili_sorgu += 1
            
            # Tüm kaliteli sonuçları birleştir
            yanit = "## 📋 CPR Protokol Bilgileri\n\n"
            yanit += f"**Soru:** {soru}\n\n"
            
            for i, sonuc in enumerate(kaliteli_sonuclar[:3]):  # İlk 3 sonuç
                yanit += f"### 🔍 Sonuç {i+1}\n"
                yanit += f"**Kategori:** {sonuc['metadata']['kategori'].replace('_', ' ').title()}\n"
                yanit += f"**Alt Kategori:** {sonuc['metadata']['alt_kategori'].title()}\n"
                yanit += f"**İçerik:** {sonuc['icerik']}\n\n"
                yanit += f"**Güvenilirlik:** {sonuc['metadata']['guvenilirlik'] * 100:.0f}% • "
                yanit += f"**Benzerlik:** {sonuc['benzerlik_skoru']:.3f} • "
                yanit += f"**Kaynak:** {sonuc['metadata']['kaynak']}\n\n"
                yanit += "---\n\n"
            
            basarili = True
        else:
            # ✅ DAHA YARDIMCI HATA MESAJI
            yanit = "## ⚠️ CPR Bilgi Merkezi\n\n"
            yanit += f"**Soru:** {soru}\n\n"
            yanit += "**Durum:** Spesifik protokol bulunamadı\n\n"
            yanit += "### 💡 Öneriler:\n"
            yanit += "• **CPR, AED, epinefrin** gibi anahtar kelimeler kullanın\n"
            yanit += "• **Yaş grubu** belirtin (yetişkin/çocuk/bebek)\n"
            yanit += "• **Daha spesifik** sorular sorun\n"
            yanit += "• **Acil durum protokolleri** için acil kelimeler kullanın\n\n"
            yanit += "### 🎯 Örnek Sorular:\n"
            yanit += "- CPR kompresyon oranı nedir?\n"
            yanit += "- Çocuklarda kalp masajı nasıl yapılır?\n"
            yanit += "- AED cihazı nasıl kullanılır?\n"
            yanit += "- Hipotermi durumunda CPR protokolü nedir?"
            
            basarili = False
        
        # Performans değerlendirmesi
        if kaliteli_sonuclar:
            en_iyi_skor = kaliteli_sonuclar[0]['benzerlik_skoru']
            if en_iyi_skor > 0.6:
                performans = "Çok Yüksek"
            elif en_iyi_skor > 0.4:
                performans = "Yüksek"
            elif en_iyi_skor > 0.3:
                performans = "Orta"
            elif en_iyi_skor > 0.2:
                performans = "Düşük"
            else:
                performans = "Çok Düşük"
        else:
            performans = "Çok Düşük"
            en_iyi_skor = 0
        
        return {
            "basarili": basarili,
            "yanit": yanit,
            "bulunan_dokuman_sayisi": len(sonuclar),
            "kaliteli_sonuc_sayisi": len(kaliteli_sonuclar),
            "en_iyi_skor": en_iyi_skor,
            "sistem_performansi": performans,
            "cpr_odakli": cpr_odakli,
            "acil_durum": acil_durum,
            "sonuc_detaylari": [
                (s['benzerlik_skoru'], s['metadata']['kategori'], s['metadata']['alt_kategori']) 
                for s in kaliteli_sonuclar[:3]
            ]
        }
class CPREgitimSistemi:
    def __init__(self):
        self.retriever = ChromaRetriever()
        self.bilgi_bankasi = []
        self.sorgu_sayisi = 0
        self.basarili_sorgu = 0
        
    def sistem_baslat(self):
        try:
            # Önce ChromaDB'yi başlat
            if not self.retriever.sistem_baslat():
                return False
            
            # JSON dosyasını oku
            try:
                with open('cpr_egitim_bilgi_bankasi.json', 'r', encoding='utf-8') as f:
                    self.bilgi_bankasi = json.load(f)
            except FileNotFoundError:
                st.error("❌ JSON dosyası bulunamadı: cpr_egitim_bilgi_bankasi.json")
                return False
            
            # JSON formatını kontrol et
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
            
            # Vector database'e dokümanları ekle (eğer boşsa)
            if self.retriever.koleksiyon.count() == 0:
                st.info("📦 Vector database'e dokümanlar yükleniyor...")
                if not self.retriever.dokumanlar_ekle(self.bilgi_bankasi, temizle=True):
                    return False
            else:
                st.success(f"✅ Vector database hazır: {self.retriever.koleksiyon.count()} doküman")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Sistem başlatma hatası: {str(e)}")
            return False
    
    def soru_sor(self, soru):
        self.sorgu_sayisi += 1
        
        # ChromaDB ile arama yap
        sonuclar = self.retriever.hizli_arama(soru, top_k=5)
        
        # CPR odaklı ve acil durum analizi
        cpr_odakli = any(terim in soru.lower() for terim in [
            "cpr", "kalp masajı", "sunî solunum", "resüsitasyon", "kalp durması", 
            "hipotermi", "hipotermik", "cardiopulmonary", "resuscitation"
        ])
        
        acil_durum = any(terim in soru.lower() for terim in [
            "acil", "ilk yardım", "hayat kurtarma", "müdahale", "112", "kritik", 
            "arrest", "acil durum", "emergency"
        ])
        
        # Kaliteli sonuçları filtrele
        kaliteli_sonuclar = [s for s in sonuclar if s['benzerlik_skoru'] > 0.3]
        
        if kaliteli_sonuclar:
            self.basarili_sorgu += 1
            
            # En iyi sonucu al
            en_iyi_sonuc = kaliteli_sonuclar[0]
            yanit = f"## 📋 {en_iyi_sonuc['metadata']['kategori'].replace('_', ' ').title()} - {en_iyi_sonuc['metadata']['alt_kategori'].title()}\n\n"
            yanit += f"**İçerik:** {en_iyi_sonuc['icerik']}\n\n"
            yanit += f"**Güvenilirlik:** {en_iyi_sonuc['metadata']['guvenilirlik'] * 100:.0f}%\n"
            yanit += f"**Acillik Seviyesi:** {en_iyi_sonuc['metadata']['acillik_seviyesi'].capitalize()}\n"
            yanit += f"**Kaynak:** {en_iyi_sonuc['metadata']['kaynak']}\n"
            yanit += f"**Benzerlik Skoru:** {en_iyi_sonuc['benzerlik_skoru']:.3f}"
            
            basarili = True
        else:
            yanit = "## ⚠️ Bilgi Bulunamadı\n\nSoru için yeterli bilgi bulunamadı. Lütfen daha spesifik bir CPR sorusu sorun veya aşağıdaki örnek sorulardan birini deneyin."
            basarili = False
        
        # Performans değerlendirmesi
        if kaliteli_sonuclar:
            en_iyi_skor = kaliteli_sonuclar[0]['benzerlik_skoru']
            if en_iyi_skor > 0.7:
                performans = "Çok Yüksek"
            elif en_iyi_skor > 0.5:
                performans = "Yüksek"
            elif en_iyi_skor > 0.3:
                performans = "Orta"
            else:
                performans = "Düşük"
        else:
            performans = "Çok Düşük"
            en_iyi_skor = 0
        
        return {
            "basarili": basarili,
            "yanit": yanit,
            "bulunan_dokuman_sayisi": len(sonuclar),
            "kaliteli_sonuc_sayisi": len(kaliteli_sonuclar),
            "en_iyi_skor": en_iyi_skor,
            "sistem_performansi": performans,
            "cpr_odakli": cpr_odakli,
            "acil_durum": acil_durum,
            "sonuc_detaylari": [
                (s['benzerlik_skoru'], s['metadata']['kategori'], s['metadata']['alt_kategori']) 
                for s in kaliteli_sonuclar[:3]
            ]
        }

# 🎨 STREAMLIT ARAYÜZ
st.set_page_config(
    page_title="CPR Eğitim Sistemi - ChromaDB",
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
    "AED nasıl kullanılır?"
]

# 🎨 SIDEBAR
with st.sidebar:
    st.header("🫀 CPR Hızlı Erişim")
    st.markdown("---")
    
    st.subheader("🚀 Örnek Sorular")
    for soru in ornek_sorular:
        if st.button(soru, key=f"btn_{soru}", use_container_width=True):
            st.session_state.soru_input = soru
    
    st.markdown("---")
    st.subheader("⚙️ Sistem Bilgisi")
    
    if CHROMA_AVAILABLE and TRANSFORMERS_AVAILABLE:
        st.success("✅ ChromaDB & Transformers Hazır")
    else:
        st.error("❌ Kütüphaneler Eksik")
        st.code("pip install chromadb sentence-transformers")
    
    st.markdown("---")
    st.info("""
    **Yeni Özellikler:**
    - 🚀 ChromaDB Vector Database
    - 🤖 AI-powered semantic search  
    - ⚡ Ultra hızlı arama
    - 🎯 Daha doğru sonuçlar
    """)

# 🎯 ANA İÇERİK
st.title("🫀 CPR Eğitim Sistemi - ChromaDB")
st.markdown("AI destekli **ultra hızlı** CPR protokol arama sistemi")

# 📊 SİSTEM DURUMU
if "sistem" not in st.session_state:
    st.session_state.sistem = CPREgitimSistemi()
    st.session_state.basladi = st.session_state.sistem.sistem_baslat()

if not st.session_state.basladi:
    st.error("""
    ❌ Sistem başlatılamadı. 
    - `cpr_egitim_bilgi_bankasi.json` dosyasını kontrol edin
    - Gerekli kütüphaneleri kurun: `pip install chromadb sentence-transformers`
    """)
else:
    st.success("✅ ChromaDB sistemi hazır! Sorularınızı sorabilirsiniz.")
    
    # İstatistikler
    if hasattr(st.session_state.sistem.retriever, 'koleksiyon'):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Toplam Bilgi", st.session_state.sistem.retriever.koleksiyon.count())
        with col2:
            st.metric("🔄 Sistem", "ChromaDB")
        with col3:
            st.metric("⭐ Teknoloji", "AI + Vector")
        with col4:
            st.metric("⚡ Hız", "Ultra Hızlı")

    st.markdown("---")
    
    # 🎯 SORU SORMA BÖLÜMÜ
    st.header("💬 Soru Sor")
    
    soru = st.text_input(
        "CPR hakkında sorunuzu yazın:",
        value=st.session_state.get('soru_input', ''),
        placeholder="Örn: CPR kompresyon oranı nedir?",
        key="soru_input_main"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sorgula_btn = st.button("🔍 AI ile Sorgula", use_container_width=True, type="primary")
    with col2:
        if st.button("🔄 Temizle", use_container_width=True):
            st.session_state.soru_input = ""
            st.rerun()
    with col3:
        if st.button("🎲 Rastgele Soru", use_container_width=True):
            st.session_state.soru_input = random.choice(ornek_sorular)
            st.rerun()
    
    # 🔍 SONUÇLARI GÖSTER
    if sorgula_btn and soru.strip():
        with st.spinner("🤖 AI sorguyu analiz ediyor..."):
            sonuc = st.session_state.sistem.soru_sor(soru)
        
        st.markdown("---")
        
        if sonuc["basarili"]:
            st.success("✅ AI uygun bilgi buldu!")
        else:
            st.warning("⚠️ Yeterli bilgi bulunamadı.")
        
        st.markdown(sonuc['yanit'])
        
        # 📊 İSTATİSTİKLER
        st.markdown("---")
        st.subheader("📊 AI Arama İstatistikleri")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**🔍 Bulunan Doküman:** {sonuc['bulunan_dokuman_sayisi']}")
            st.info(f"**⭐ Kaliteli Sonuç:** {sonuc['kaliteli_sonuc_sayisi']}")
            st.info(f"**🏆 En İyi Skor:** {sonuc['en_iyi_skor']:.3f}")
        with col2:
            st.info(f"**📈 Performans:** {sonuc['sistem_performansi']}")
            st.info(f"**💡 CPR Odaklı:** {'✅ Evet' if sonuc['cpr_odakli'] else '❌ Hayır'}")
            st.info(f"**🚨 Acil Durum:** {'✅ Evet' if sonuc['acil_durum'] else '❌ Hayır'}")
        
        # 📑 DETAYLI SONUÇLAR
        if sonuc["sonuc_detaylari"]:
            st.markdown("---")
            st.subheader("📑 En İyi Eşleşmeler")
            
            tablo_verisi = {
                "🏆 Skor": [f"{s[0]:.3f}" for s in sonuc["sonuc_detaylari"]],
                "📂 Kategori": [s[1] for s in sonuc["sonuc_detaylari"]],
                "📁 Alt Kategori": [s[2] for s in sonuc["sonuc_detaylari"]]
            }
            
            st.dataframe(tablo_verisi, use_container_width=True, hide_index=True)
    
    elif sorgula_btn and not soru.strip():
        st.warning("❗ Lütfen bir soru giriniz.")

# 📱 FOOTER
st.markdown("---")
st.caption("""
🚀 **ChromaDB AI Sistemi** - Vector database + Sentence transformers  
⚡ **Ultra Hızlı** - Gerçek zamanlı semantik arama  
🎯 **Akıllı** - Anlamsal benzerlik ile arama  
🔒 **Güvenli** - Yerel işlem, veri paylaşımı yok
""")