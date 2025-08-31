# ✅ DÜZELTİLMİŞ HIZLI CPR SİSTEMİ
# Bu kodu cpr_fast_fixed.py olarak kaydet

import streamlit as st
import json
import numpy as np
import random
import time
import uuid
from typing import List, Dict

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class HizliRetriever:
    def __init__(self):
        self.koleksiyon_adi = "cpr_hizli_db"
        self.chroma_client = None
        self.koleksiyon = None
        self.model = None
        
        # 🎯 TÜRKÇE KELİME EŞLEŞTİRME SÖZLÜĞÜ
        self.kelime_eslestirme = {
            'epinefrin': ['epinephrine', 'adrenaline', 'adrenalin', 'vazopresor'],
            'amiodarone': ['amiodaron', 'antiaritmik', 'kardiak'],
            'cpr': ['kalp masajı', 'resüsitasyon', 'canlandırma', 'compression'],
            'aed': ['defibrillatör', 'şok', 'defibrillation'],
            'doz': ['dose', 'miktar', 'amount', 'dosage', 'mg'],
            'yetişkin': ['adult', 'erişkin', 'büyük'],
            'çocuk': ['child', 'pediatric', 'küçük'],
            'bebek': ['infant', 'baby'],
            'hipotermik': ['hypothermic', 'soğuk', 'cold'],
            'kompresyon': ['compression', 'basınç', 'göğüs basısı'],
            'oran': ['ratio', 'rate', 'frequency'],
            'derinlik': ['depth', 'profundidad']
        }
    
    def sistem_baslat(self):
        if not CHROMA_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            st.error("❌ pip install chromadb sentence-transformers")
            return False
        
        try:
            with st.spinner("⚡ Hızlı AI sistemi başlatılıyor..."):
                # ChromaDB
                self.chroma_client = chromadb.Client(Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                ))
                
                # 🚀 KÜÇÜK VE HIZLI MODEL - 80MB
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                
                try:
                    self.koleksiyon = self.chroma_client.get_collection(self.koleksiyon_adi)
                    st.success(f"✅ Hızlı database hazır: {self.koleksiyon.count()} doküman")
                except:
                    self.koleksiyon = self.chroma_client.create_collection(
                        name=self.koleksiyon_adi,
                        metadata={"version": "fast", "model": "MiniLM"}
                    )
                    st.info("🆕 Hızlı database oluşturuluyor...")
                
            return True
            
        except Exception as e:
            st.error(f"❌ Sistem hatası: {str(e)}")
            return False
    
    def _kelime_genislet(self, metin: str) -> str:
        """Türkçe kelime genişletme"""
        genisletilmis = metin.lower()
        
        for anahtar, esanlamlilar in self.kelime_eslestirme.items():
            if anahtar in genisletilmis:
                # Eşanlamlıları ekle
                genisletilmis += " " + " ".join(esanlamlilar)
            
            # Ters kontrol - eşanlamlılardan biri varsa anahtar kelimeyi ekle
            for esanlamli in esanlamlilar:
                if esanlamli in genisletilmis and anahtar not in genisletilmis:
                    genisletilmis += " " + anahtar
        
        return genisletilmis
    
    def dokumanlar_ekle(self, dokumanlar: List[Dict], temizle: bool = False):
        if not self.koleksiyon or not self.model:
            return False
        
        if temizle:
            try:
                self.chroma_client.delete_collection(self.koleksiyon_adi)
            except:
                pass
            self.koleksiyon = self.chroma_client.create_collection(
                name=self.koleksiyon_adi,
                metadata={"version": "fast", "model": "MiniLM"}
            )
        
        progress = st.progress(0)
        status = st.empty()
        
        ids, embeddings, metadatas, documents = [], [], [], []
        
        for i, dok in enumerate(dokumanlar):
            ids.append(dok.get('id', str(uuid.uuid4())))
            
            # 🎯 SÜPER KELİME GENİŞLETME
            temel_icerik = dok['icerik']
            kategori = dok.get('kategori', '')
            alt_kategori = dok.get('alt_kategori', '')
            
            # Tam içerik genişletme
            icerik_genisletilmis = self._kelime_genislet(
                f"{temel_icerik} {kategori} {alt_kategori}"
            )
            
            # Hızlı embedding oluştur
            embedding = self.model.encode(icerik_genisletilmis).tolist()
            embeddings.append(embedding)
            
            # Metadata
            metadatas.append({
                'kategori': kategori,
                'alt_kategori': alt_kategori,
                'guvenilirlik': float(dok.get('guvenilirlik', 0.8)),
                'acillik_seviyesi': dok.get('acillik_seviyesi', 'normal'),
                'kaynak': dok.get('metadata', {}).get('kaynak', 'AHA Guidelines')
            })
            
            documents.append(temel_icerik)
            
            # Progress
            if i % 2 == 0:
                progress.progress((i + 1) / len(dokumanlar))
                status.text(f"⚡ İşleniyor: {i + 1}/{len(dokumanlar)}")
        
        try:
            self.koleksiyon.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
            progress.progress(1.0)
            status.success(f"✅ {len(dokumanlar)} doküman eklendi!")
            time.sleep(1)
            progress.empty()
            status.empty()
            return True
            
        except Exception as e:
            st.error(f"❌ Ekleme hatası: {str(e)}")
            return False
    
    def hizli_arama(self, sorgu: str, top_k: int = 8) -> List[Dict]:
        if not self.koleksiyon or not self.model:
            return []
        
        try:
            # 🎯 SORGUYU GENİŞLET
            genisletilmis_sorgu = self._kelime_genislet(sorgu)
            
            print(f"[DEBUG] Orijinal: '{sorgu}'")
            print(f"[DEBUG] Genişletilmiş: '{genisletilmis_sorgu}'")
            
            # Hızlı embedding ve arama
            sorgu_embedding = self.model.encode(genisletilmis_sorgu).tolist()
            
            sonuclar = self.koleksiyon.query(
                query_embeddings=[sorgu_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Sonuçları işle
            arama_sonuclari = []
            if sonuclar['documents'] and len(sonuclar['documents'][0]) > 0:
                for i in range(len(sonuclar['documents'][0])):
                    distance = sonuclar['distances'][0][i]
                    similarity = max(0.0, 1.0 - distance)
                    
                    # Güvenilirlik bonusu
                    guvenilirlik = sonuclar['metadatas'][0][i].get('guvenilirlik', 0.8)
                    final_score = similarity * (0.8 + 0.2 * guvenilirlik)
                    
                    arama_sonuclari.append({
                        'id': sonuclar['ids'][0][i],
                        'icerik': sonuclar['documents'][0][i],
                        'benzerlik_skoru': final_score,
                        'metadata': sonuclar['metadatas'][0][i],
                        'kategori': sonuclar['metadatas'][0][i].get('kategori', 'genel'),
                        'guvenilirlik': guvenilirlik
                    })
            
            # Skor bazında sırala
            arama_sonuclari.sort(key=lambda x: x['benzerlik_skoru'], reverse=True)
            
            # Debug bilgisi
            print(f"[HIZLI DEBUG] '{sorgu}' için {len(arama_sonuclari)} sonuç:")
            for i, sonuc in enumerate(arama_sonuclari[:3]):
                print(f"  {i+1}. Skor: {sonuc['benzerlik_skoru']:.3f}, Kategori: {sonuc['kategori']}")
            
            return arama_sonuclari
            
        except Exception as e:
            st.error(f"❌ Arama hatası: {str(e)}")
            return []

class HizliCPRSistemi:
    def __init__(self):
        self.retriever = HizliRetriever()
        self.bilgi_bankasi = []
        self.toplam_sorgu = 0
        self.basarili_sorgu = 0
    
    def sistem_baslat(self):
        try:
            if not self.retriever.sistem_baslat():
                return False
            
            # JSON yükle
            try:
                with open('cpr_egitim_bilgi_bankasi.json', 'r', encoding='utf-8') as f:
                    self.bilgi_bankasi = json.load(f)
            except FileNotFoundError:
                st.error("❌ cpr_egitim_bilgi_bankasi.json bulunamadı!")
                return False
            
            # Database'e yükle
            if self.retriever.koleksiyon.count() == 0:
                st.info("⚡ Hızlı database yükleniyor...")
                if not self.retriever.dokumanlar_ekle(self.bilgi_bankasi, temizle=True):
                    return False
            
            return True
            
        except Exception as e:
            st.error(f"❌ Sistem hatası: {str(e)}")
            return False
    
    def hizli_sorgulama(self, soru: str) -> Dict:
        self.toplam_sorgu += 1
        
        # Hızlı arama
        sonuclar = self.retriever.hizli_arama(soru, top_k=6)
        
        # 🎯 DİNAMİK EŞİK DEĞERLERİ
        if 'doz' in soru.lower() or 'miktar' in soru.lower() or 'mg' in soru.lower():
            esik = 0.05  # Doz soruları için çok düşük
        elif 'acil' in soru.lower() or 'kritik' in soru.lower():
            esik = 0.03  # Acil için ultra düşük
        elif 'nasıl' in soru.lower() or 'how' in soru.lower():
            esik = 0.08  # Prosedür soruları
        else:
            esik = 0.10  # Normal sorular
        
        kaliteli_sonuclar = [s for s in sonuclar if s['benzerlik_skoru'] > esik]
        
        # CPR analizi
        cpr_kelimeler = ['cpr', 'kalp', 'masaj', 'resüsitasyon', 'aed', 'epinefrin', 'kompresyon']
        cpr_odakli = any(kelime in soru.lower() for kelime in cpr_kelimeler)
        
        # Yanıt oluştur
        if len(kaliteli_sonuclar) >= 1:
            self.basarili_sorgu += 1
            yanit = self._hizli_yanit_olustur(soru, kaliteli_sonuclar[:2])
            basarili = True
        else:
            yanit = self._hizli_oneri_sistemi(soru)
            basarili = False
        
        # Performans
        if kaliteli_sonuclar:
            en_iyi_skor = kaliteli_sonuclar[0]['benzerlik_skoru']
            if en_iyi_skor > 0.4:
                performans = "🚀 Mükemmel"
            elif en_iyi_skor > 0.2:
                performans = "📈 Çok İyi"
            elif en_iyi_skor > 0.1:
                performans = "📊 İyi"
            else:
                performans = "📉 Orta"
        else:
            en_iyi_skor = 0
            performans = "⚠️ Düşük"
        
        return {
            "basarili": basarili,
            "yanit": yanit,
            "bulunan_dokuman_sayisi": len(sonuclar),
            "kaliteli_sonuc_sayisi": len(kaliteli_sonuclar),
            "en_iyi_skor": en_iyi_skor,
            "sistem_performansi": performans,
            "cpr_odakli": cpr_odakli,
            "kullanilan_esik": esik,
            "basari_orani": f"{(self.basarili_sorgu/max(1,self.toplam_sorgu))*100:.1f}%",
            "sonuc_detaylari": [
                (s['benzerlik_skoru'], s['metadata']['kategori'], s['metadata']['alt_kategori']) 
                for s in kaliteli_sonuclar[:2]
            ]
        }
    
    def _hizli_yanit_olustur(self, soru: str, sonuclar: List[Dict]) -> str:
        yanit = f"## ⚡ HIZLI CPR REHBERİ\n\n**Soru:** {soru}\n\n"
        
        for i, sonuc in enumerate(sonuclar):
            yanit += f"### 📌 Protokol {i+1}\n"
            yanit += f"**Kategori:** {sonuc['metadata']['kategori'].replace('_', ' ').title()}\n"
            yanit += f"**Alt Kategori:** {sonuc['metadata']['alt_kategori'].replace('_', ' ').title()}\n"
            yanit += f"**İçerik:** {sonuc['icerik']}\n\n"
            
            # Kalite göstergeleri
            yildiz_sayisi = min(5, int(sonuc['benzerlik_skoru'] * 10))
            yildizlar = "⭐" * max(1, yildiz_sayisi)
            
            yanit += f"**Kalite:** {yildizlar} ({sonuc['benzerlik_skoru']:.3f}) • "
            yanit += f"**Güvenilirlik:** %{sonuc['metadata']['guvenilirlik']*100:.0f} • "
            yanit += f"**Kaynak:** {sonuc['metadata']['kaynak']}\n\n"
            yanit += "---\n\n"
        
        yanit += "### ⚕️ HIZLI UYARI\n"
        yanit += "• **AHA 2020 Guidelines** temelinde\n"
        yanit += "• **Acil durumlar:** 112\n"
        yanit += "• **Eğitim amaçlı** kullanın\n"
        
        return yanit
    
    def _hizli_oneri_sistemi(self, soru: str) -> str:
        return f"""## 🎯 HIZLI ÖNERİLER

**Soru:** {soru}

**Durum:** Spesifik bilgi bulunamadı, öneriler sunuluyor.

### 💡 Arama Önerileri:
• **Daha spesifik olun:** "Epinefrin dozu kaç mg?"
• **Sayısal sorular:** "CPR oranı kaçtır?"
• **Yaş grubu belirtin:** "Çocuklarda", "Yetişkinlerde"
• **Protokol adı:** AED, CPR, defibrilasyon

### 🔍 Popüler Sorular:
- Epinefrin dozu nedir?
- CPR kompresyon oranı kaçtır?
- AED nasıl kullanılır?
- Çocuk CPR protokolü nedir?

### 📞 Acil Durumlar: 112
"""

# 🎨 STREAMLIT ARAYÜZÜ - HIZLI VE BASIT
st.set_page_config(
    page_title="CPR Hızlı Sistem ⚡",
    page_icon="🫀",
    layout="wide"
)

# CSS - Basit ve hızlı
st.markdown("""
<style>
.hizli-kart {
    background: linear-gradient(45deg, #ff6b6b, #ffa500);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🫀 CPR Hızlı Eğitim Sistemi ⚡")
st.markdown("**Model boyutu: sadece 80MB** - Ultra hızlı yanıt garantisi")

# Hızlı örnek sorular - sidebar
with st.sidebar:
    st.markdown("## ⚡ Hızlı Sorular")
    
    hizli_ornekler = [
        "Epinefrin dozu nedir?",
        "CPR kompresyon oranı?", 
        "AED nasıl kullanılır?",
        "Çocuk CPR protokolü?",
        "Yetişkin CPR derinliği?",
        "Amiodarone dozu?"
    ]
    
    for ornek in hizli_ornekler:
        if st.button(ornek, key=f"hizli_{ornek}", width='stretch'):
            st.session_state.hizli_soru = ornek
    
    st.markdown("---")
    st.info("""
    🚀 **Hızlı Özellikler:**
    • 80MB küçük model
    • 2-3 saniye yanıt
    • Türkçe kelime genişletme
    • Dinamik eşik değerleri
    """)

# Sistem başlatma
if "hizli_sistem" not in st.session_state:
    st.session_state.hizli_sistem = HizliCPRSistemi()
    st.session_state.hizli_basladi = st.session_state.hizli_sistem.sistem_baslat()

if st.session_state.hizli_basladi:
    st.markdown('<div class="hizli-kart">✅ Hızlı sistem hazır! Model: MiniLM (80MB)</div>', unsafe_allow_html=True)
    
    # Metrikler
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Database", st.session_state.hizli_sistem.retriever.koleksiyon.count())
    with col2:
        st.metric("🚀 Model", "MiniLM")
    with col3:
        st.metric("💾 Boyut", "80MB")
    with col4:
        basari = f"{(st.session_state.hizli_sistem.basarili_sorgu/max(1,st.session_state.hizli_sistem.toplam_sorgu))*100:.1f}%"
        st.metric("📈 Başarı", basari)
    
    st.markdown("---")
    
    # Ana soru bölümü
    st.markdown("## 💬 Hızlı Sorgulama")
    
    soru = st.text_input(
        "CPR sorunuzu yazın:",
        value=st.session_state.get('hizli_soru', ''),
        placeholder="Örn: Epinefrin dozu kaç mg olmalıdır?",
        key="ana_soru_input"
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        ara_btn = st.button("⚡ Hızlı Ara", type="primary", width='stretch')
    with col2:
        if st.button("🔄 Temizle", width='stretch'):
            st.session_state.hizli_soru = ""
            st.rerun()
    with col3:
        if st.button("🎲 Rastgele", width='stretch'):
            st.session_state.hizli_soru = random.choice(hizli_ornekler)
            st.rerun()
    
    # SONUÇLARI GÖSTER
    if ara_btn and soru.strip():
        with st.spinner("⚡ Hızlı AI analizi..."):
            time.sleep(0.3)  # Hızlı his için
            sonuc = st.session_state.hizli_sistem.hizli_sorgulama(soru)
        
        st.markdown("---")
        
        if sonuc["basarili"]:
            st.success("✅ Hızlı protokol bulundu!")
        else:
            st.warning("⚠️ Hızlı öneriler sunuluyor")
        
        # Ana yanıt
        st.markdown(sonuc['yanit'])
        
        # Hızlı istatistikler
        st.markdown("---")
        st.markdown("### 📊 Hızlı Analiz Sonuçları")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**🎯 En İyi Skor:** {sonuc['en_iyi_skor']:.3f}")
            st.info(f"**📊 Performans:** {sonuc['sistem_performansi']}")
        with col2:
            st.info(f"**🔍 Bulunan:** {sonuc['bulunan_dokuman_sayisi']}")
            st.info(f"**⭐ Kaliteli:** {sonuc['kaliteli_sonuc_sayisi']}")
        with col3:
            st.info(f"**🎚️ Eşik:** {sonuc['kullanilan_esik']:.3f}")
            st.info(f"**📈 Başarı:** {sonuc['basari_orani']}")
        
        # Detaylı sonuçlar tablosu
        if sonuc["sonuc_detaylari"]:
            st.markdown("---")
            st.markdown("### 🎯 En İyi Eşleşmeler")
            
            tablo_data = []
            for skor, kategori, alt_kategori in sonuc["sonuc_detaylari"]:
                tablo_data.append({
                    "🏆 Skor": f"{skor:.3f}",
                    "📂 Kategori": kategori.replace('_', ' ').title(),
                    "🏷️ Alt Kategori": alt_kategori.replace('_', ' ').title()
                })
            
            st.dataframe(tablo_data, use_container_width=True, hide_index=True)
    
    elif ara_btn and not soru.strip():
        st.warning("❗ Lütfen bir CPR sorusu yazın")

else:
    st.error("❌ Hızlı sistem başlatılamadı - JSON dosyasını kontrol edin")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white;">
    <h4>⚡ CPR Hızlı Eğitim Sistemi</h4>
    <p><strong>80MB Model - Ultra Hızlı Yanıt - Türkçe Optimizasyonu</strong></p>
    <p>🎯 Kelime Genişletme • 📊 Dinamik Eşikler • ⚡ 2-3sn Yanıt</p>
</div>
""", unsafe_allow_html=True)

st.caption("⚠️ Eğitim amaçlıdır. Acil durumlarda 112'yi arayın.")""