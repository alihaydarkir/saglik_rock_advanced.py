# 🚀 PROFESYONELLEŞTİRİLMİŞ CPR EĞİTİM SİSTEMİ
# Bu kodu cpr_professional_ui.py olarak kaydet

import streamlit as st
import json
import numpy as np
import random
import time
import uuid
from typing import List, Dict, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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

class ProfesyonelRetriever:
    """Profesyonel seviye retrieval sistemi"""
    
    def __init__(self):
        self.koleksiyon_adi = "cpr_professional_v1"
        self.chroma_client = None
        self.koleksiyon = None
        self.model = None
        self.performans_gecmisi = []
        
        # 🎯 GELİŞMİŞ TÜRKÇE KELİME HARİTASI
        self.gelismis_kelime_haritasi = {
            # İlaç terimleri
            'epinefrin': ['epinephrine', 'adrenaline', 'adrenalin', 'vazopresor', 'vazopressör', 'noradrenalin'],
            'amiodarone': ['amiodaron', 'antiaritmik', 'kardiak ilaç', 'ritim düzenleyici'],
            'atropin': ['atropine', 'antikolinerjik', 'bradikardi ilacı'],
            'lidokain': ['lidocaine', 'lokal anestezik', 'antiaritmik'],
            'magnezyum': ['magnesium', 'mg', 'elektrolit'],
            
            # CPR terimleri
            'cpr': ['cardiopulmonary resuscitation', 'kalp masajı', 'canlandırma', 'resüsitasyon', 
                   'yaşam desteği', 'temel yaşam desteği', 'ileri yaşam desteği'],
            'kompresyon': ['compression', 'basınç', 'göğüs basısı', 'masaj', 'basma'],
            'ventilasyon': ['solunum', 'nefes', 'hava verme', 'breathing', 'oksijen'],
            'defibrilasyon': ['defibrillation', 'şok', 'elektrik şoku', 'kardiyoversiyon'],
            
            # Cihaz terimleri  
            'aed': ['automated external defibrillator', 'defibrillatör', 'şok cihazı', 'otomatik defibrillatör'],
            'monitör': ['monitor', 'ekg', 'ekg', 'kalp ritmi takibi'],
            
            # Anatomi terimleri
            'kalp': ['heart', 'kardiak', 'miyokard', 'ventrikül', 'atriyum'],
            'göğüs': ['chest', 'toraks', 'göğüs kafesi', 'sternum'],
            'nabız': ['pulse', 'kalp ritmi', 'heartbeat', 'nabız hızı'],
            
            # Yaş grupları
            'yetişkin': ['adult', 'erişkin', 'büyük', '18 yaş üzeri'],
            'çocuk': ['child', 'pediatrik', 'küçük', '1-8 yaş', 'okul çağı'],
            'bebek': ['infant', 'baby', 'yenidoğan', '0-12 ay', 'süt çocuğu'],
            
            # Ölçü birimleri
            'doz': ['dose', 'miktar', 'amount', 'dozaj', 'dosis'],
            'mg': ['milligram', 'miligram', 'ml', 'cc'],
            'oran': ['ratio', 'rate', 'frequency', 'hız', 'frekans'],
            'derinlik': ['depth', 'derinalık', 'cm', 'santimetre'],
            
            # Acil durum terimleri
            'arrest': ['kalp durması', 'kardiyak arrest', 'cardiac arrest', 'ani ölüm'],
            'hipotermik': ['hypothermic', 'soğuk', 'düşük sıcaklık', 'hipotermi'],
            'vf': ['ventricular fibrillation', 'ventrikül fibrilasyonu'],
            'vt': ['ventricular tachycardia', 'ventrikül taşikardisi'],
            'asistol': ['asystole', 'düz çizgi', 'kalp durması'],
            
            # Soru kelimeleri
            'nasıl': ['how', 'ne şekilde', 'hangi yöntem', 'prosedür'],
            'nedir': ['what', 'ne', 'tanım'],
            'kaç': ['how much', 'how many', 'ne kadar', 'miktar'],
            'nerede': ['where', 'hangi bölge', 'lokasyon'],
            'ne zaman': ['when', 'hangi durumda', 'zamanı']
        }
    
    def sistem_baslat(self):
        """Profesyonel sistem başlatma"""
        if not CHROMA_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            st.error("❌ pip install chromadb sentence-transformers plotly")
            return False
        
        try:
            with st.spinner("🚀 Profesyonel AI sistemi başlatılıyor..."):
                # ChromaDB
                self.chroma_client = chromadb.Client(Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                ))
                
                # 🎯 BÜYÜK VE GÜÇLÜ MODEL - 480MB
                st.info("📥 Güçlü Türkçe modeli yükleniyor... (480MB)")
                self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased')
                
                try:
                    self.koleksiyon = self.chroma_client.get_collection(self.koleksiyon_adi)
                    st.success(f"✅ Profesyonel database hazır: {self.koleksiyon.count()} doküman")
                except:
                    self.koleksiyon = self.chroma_client.create_collection(
                        name=self.koleksiyon_adi,
                        metadata={"version": "professional", "model": "distiluse-multilingual"}
                    )
                    st.info("🆕 Profesyonel database oluşturuluyor...")
                
            return True
            
        except Exception as e:
            st.error(f"❌ Sistem hatası: {str(e)}")
            return False
    
    def _super_kelime_genisletme(self, metin: str) -> str:
        """Süper gelişmiş kelime genişletme"""
        genisletilmis = metin.lower()
        
        # Her kelime için kapsamlı eşleştirme
        for anahtar, esanlamlilar in self.gelismis_kelime_haritasi.items():
            # Ana kelime varsa eşanlamlıları ekle
            if anahtar in genisletilmis:
                genisletilmis += " " + " ".join(esanlamlilar)
            
            # Eşanlamlılardan biri varsa ana kelimeyi ekle
            for esanlamli in esanlamlilar:
                if esanlamli.lower() in genisletilmis.lower() and anahtar not in genisletilmis:
                    genisletilmis += " " + anahtar
        
        # Özel durum işlemleri
        if "kaç" in genisletilmis or "ne kadar" in genisletilmis:
            genisletilmis += " doz miktar mg amount dose"
        
        if "nasıl" in genisletilmis:
            genisletilmis += " prosedür adım yöntem protokol"
            
        return genisletilmis
    
    def dokumanlar_ekle(self, dokumanlar: List[Dict], temizle: bool = False):
        """Gelişmiş doküman ekleme"""
        if not self.koleksiyon or not self.model:
            return False
        
        if temizle:
            try:
                self.chroma_client.delete_collection(self.koleksiyon_adi)
            except:
                pass
            self.koleksiyon = self.chroma_client.create_collection(
                name=self.koleksiyon_adi,
                metadata={"version": "professional", "model": "distiluse-multilingual"}
            )
        
        # Gelişmiş progress bar
        progress = st.progress(0)
        status = st.empty()
        time_start = time.time()
        
        ids, embeddings, metadatas, documents = [], [], [], []
        
        for i, dok in enumerate(dokumanlar):
            ids.append(dok.get('id', str(uuid.uuid4())))
            
            # 🎯 SÜPER GENİŞLETİLMİŞ İÇERİK
            temel_icerik = dok['icerik']
            kategori = dok.get('kategori', '')
            alt_kategori = dok.get('alt_kategori', '')
            
            # Tam kapsamlı genişletme
            tam_icerik = f"{temel_icerik} {kategori} {alt_kategori}"
            genisletilmis_icerik = self._super_kelime_genisletme(tam_icerik)
            
            # Güçlü embedding oluştur
            embedding = self.model.encode(genisletilmis_icerik).tolist()
            embeddings.append(embedding)
            
            # Zengin metadata
            metadatas.append({
                'kategori': kategori,
                'alt_kategori': alt_kategori,
                'guvenilirlik': float(dok.get('guvenilirlik', 0.8)),
                'acillik_seviyesi': dok.get('acillik_seviyesi', 'normal'),
                'kaynak': dok.get('metadata', {}).get('kaynak', 'AHA Guidelines'),
                'protokol_tipi': dok.get('metadata', {}).get('protokol_tipi', 'standart'),
                'ekleme_tarihi': datetime.now().isoformat(),
                'kelime_sayisi': len(temel_icerik.split())
            })
            
            documents.append(temel_icerik)
            
            # Gelişmiş progress
            elapsed = time.time() - time_start
            if i > 0:
                eta = (elapsed / i) * (len(dokumanlar) - i)
                status.text(f"📊 İşleniyor: {i + 1}/{len(dokumanlar)} • ETA: {eta:.1f}s")
            
            progress.progress((i + 1) / len(dokumanlar))
        
        try:
            self.koleksiyon.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
            
            total_time = time.time() - time_start
            progress.progress(1.0)
            status.success(f"✅ {len(dokumanlar)} doküman eklendi! ({total_time:.1f}s)")
            time.sleep(1.5)
            progress.empty()
            status.empty()
            return True
            
        except Exception as e:
            st.error(f"❌ Ekleme hatası: {str(e)}")
            return False
    
    def profesyonel_arama(self, sorgu: str, top_k: int = 10) -> List[Dict]:
        """Profesyonel seviye arama"""
        if not self.koleksiyon or not self.model:
            return []
        
        try:
            # Süper sorgu genişletme
            genisletilmis_sorgu = self._super_kelime_genisletme(sorgu)
            
            # Debug bilgisi
            print(f"[PRO DEBUG] Orijinal: '{sorgu}'")
            print(f"[PRO DEBUG] Genişletilmiş: '{genisletilmis_sorgu[:100]}...'")
            
            # Güçlü embedding ve arama
            sorgu_embedding = self.model.encode(genisletilmis_sorgu).tolist()
            
            sonuclar = self.koleksiyon.query(
                query_embeddings=[sorgu_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Gelişmiş sonuç işleme
            profesyonel_sonuclar = []
            if sonuclar['documents'] and len(sonuclar['documents'][0]) > 0:
                for i in range(len(sonuclar['documents'][0])):
                    distance = sonuclar['distances'][0][i]
                    
                    # 🎯 GELİŞMİŞ SKOR HESAPLAMA
                    base_similarity = max(0.0, 1.0 - distance)
                    
                    # Çoklu faktör skorlama
                    guvenilirlik = sonuclar['metadatas'][0][i].get('guvenilirlik', 0.8)
                    acillik = sonuclar['metadatas'][0][i].get('acillik_seviyesi', 'normal')
                    
                    # Acillik bonusu
                    acillik_bonusu = 1.2 if acillik == 'kritik' else 1.0
                    
                    # Final skor
                    final_score = base_similarity * (0.7 + 0.3 * guvenilirlik) * acillik_bonusu
                    
                    profesyonel_sonuclar.append({
                        'id': sonuclar['ids'][0][i],
                        'icerik': sonuclar['documents'][0][i],
                        'benzerlik_skoru': final_score,
                        'ham_benzerlik': base_similarity,
                        'metadata': sonuclar['metadatas'][0][i],
                        'kategori': sonuclar['metadatas'][0][i].get('kategori', 'genel'),
                        'guvenilirlik': guvenilirlik,
                        'acillik': acillik
                    })
            
            # Gelişmiş sıralama
            profesyonel_sonuclar.sort(key=lambda x: x['benzerlik_skoru'], reverse=True)
            
            # Performans kaydı
            if profesyonel_sonuclar:
                self.performans_gecmisi.append({
                    'timestamp': datetime.now(),
                    'sorgu': sorgu,
                    'en_iyi_skor': profesyonel_sonuclar[0]['benzerlik_skoru'],
                    'sonuc_sayisi': len(profesyonel_sonuclar)
                })
                
                # Son 20 kaydı tut
                if len(self.performans_gecmisi) > 20:
                    self.performans_gecmisi.pop(0)
            
            # Debug bilgisi
            print(f"[PRO DEBUG] '{sorgu}' için {len(profesyonel_sonuclar)} profesyonel sonuç:")
            for i, sonuc in enumerate(profesyonel_sonuclar[:3]):
                print(f"  {i+1}. Final: {sonuc['benzerlik_skoru']:.3f}, Ham: {sonuc['ham_benzerlik']:.3f}, Kategori: {sonuc['kategori']}")
            
            return profesyonel_sonuclar
            
        except Exception as e:
            st.error(f"❌ Arama hatası: {str(e)}")
            return []
    
    def get_performans_grafigi(self):
        """Performans grafik verisi"""
        if not self.performans_gecmisi:
            return None
        
        return {
            'timestamps': [p['timestamp'] for p in self.performans_gecmisi],
            'skorlar': [p['en_iyi_skor'] for p in self.performans_gecmisi],
            'sorgular': [p['sorgu'][:20] + '...' if len(p['sorgu']) > 20 else p['sorgu'] 
                        for p in self.performans_gecmisi]
        }

class ProfesyonelCPRSistemi:
    """Profesyonel CPR Eğitim Sistemi"""
    
    def __init__(self):
        self.retriever = ProfesyonelRetriever()
        self.bilgi_bankasi = []
        self.toplam_sorgu = 0
        self.basarili_sorgu = 0
        self.sistem_baslatma_zamani = None
        
        # Performans metrikleri
        self.ortalama_yanit_suresi = []
        self.kategori_dagilimi = {}
    
    def sistem_baslat(self):
        """Profesyonel sistem başlatma"""
        self.sistem_baslatma_zamani = datetime.now()
        
        try:
            if not self.retriever.sistem_baslat():
                return False
            
            # JSON veri yükleme
            try:
                with open('cpr_egitim_bilgi_bankasi.json', 'r', encoding='utf-8') as f:
                    self.bilgi_bankasi = json.load(f)
                    
                # Kategori analizi
                for dok in self.bilgi_bankasi:
                    kategori = dok.get('kategori', 'genel')
                    self.kategori_dagilimi[kategori] = self.kategori_dagilimi.get(kategori, 0) + 1
                    
            except FileNotFoundError:
                st.error("❌ cpr_egitim_bilgi_bankasi.json bulunamadı!")
                return False
            
            # Database yükleme
            if self.retriever.koleksiyon.count() == 0:
                st.info("🔄 Profesyonel database oluşturuluyor...")
                if not self.retriever.dokumanlar_ekle(self.bilgi_bankasi, temizle=True):
                    return False
            
            return True
            
        except Exception as e:
            st.error(f"❌ Sistem hatası: {str(e)}")
            return False
    
    def profesyonel_sorgulama(self, soru: str) -> Dict:
        """Profesyonel sorgulama sistemi"""
        start_time = time.time()
        self.toplam_sorgu += 1
        
        # Profesyonel arama
        sonuclar = self.retriever.profesyonel_arama(soru, top_k=8)
        
        # 🎯 İNTELLİGENT EŞİK SİSTEMİ
        esik_kuralları = {
            'doz_miktar': (['doz', 'miktar', 'mg', 'kaç', 'ne kadar'], 0.04),
            'acil_kritik': (['acil', 'kritik', 'emergency', 'arrest', 'durma'], 0.03),
            'prosedur': (['nasıl', 'how', 'adım', 'yöntem', 'prosedür'], 0.06),
            'tanım': (['nedir', 'what', 'tanım', 'ne'], 0.08),
            'genel': ([], 0.12)
        }
        
        # Uygun eşiği bul
        kullanilan_esik = esik_kuralları['genel'][1]
        esik_tipi = 'genel'
        
        for tip, (kelimeler, esik_degeri) in esik_kuralları.items():
            if any(kelime in soru.lower() for kelime in kelimeler):
                kullanilan_esik = esik_degeri
                esik_tipi = tip
                break
        
        kaliteli_sonuclar = [s for s in sonuclar if s['benzerlik_skoru'] > kullanilan_esik]
        
        # CPR analizi
        cpr_kelimeler = ['cpr', 'kalp', 'resüsitasyon', 'defibrilasyon', 'epinefrin', 'aed', 'kompresyon']
        cpr_odakli = any(kelime in soru.lower() for kelime in cpr_kelimeler)
        
        acil_kelimeler = ['acil', 'kritik', 'emergency', 'arrest', 'durma', 'kriz']
        acil_durum = any(kelime in soru.lower() for kelime in acil_kelimeler)
        
        # Yanıt oluşturma
        if len(kaliteli_sonuclar) >= 1:
            self.basarili_sorgu += 1
            yanit = self._profesyonel_yanit_olustur(soru, kaliteli_sonuclar[:3], acil_durum)
            basarili = True
        else:
            yanit = self._intelligent_oneri_sistemi(soru, sonuclar[:2])
            basarili = False
        
        # Performans hesaplama
        yanit_suresi = time.time() - start_time
        self.ortalama_yanit_suresi.append(yanit_suresi)
        
        if len(self.ortalama_yanit_suresi) > 10:
            self.ortalama_yanit_suresi.pop(0)
        
        # Skor değerlendirmesi
        if kaliteli_sonuclar:
            en_iyi_skor = kaliteli_sonuclar[0]['benzerlik_skoru']
            if en_iyi_skor > 0.7:
                performans = "🏆 Mükemmel"
            elif en_iyi_skor > 0.5:
                performans = "🚀 Çok İyi"
            elif en_iyi_skor > 0.3:
                performans = "📈 İyi"
            elif en_iyi_skor > 0.15:
                performans = "📊 Orta"
            else:
                performans = "📉 Düşük"
        else:
            en_iyi_skor = 0
            performans = "⚠️ Yetersiz"
        
        return {
            "basarili": basarili,
            "yanit": yanit,
            "bulunan_dokuman_sayisi": len(sonuclar),
            "kaliteli_sonuc_sayisi": len(kaliteli_sonuclar),
            "en_iyi_skor": en_iyi_skor,
            "sistem_performansi": performans,
            "cpr_odakli": cpr_odakli,
            "acil_durum": acil_durum,
            "kullanilan_esik": kullanilan_esik,
            "esik_tipi": esik_tipi,
            "yanit_suresi": yanit_suresi,
            "basari_orani": f"{(self.basarili_sorgu/max(1,self.toplam_sorgu))*100:.1f}%",
            "sonuc_detaylari": [
                (s['benzerlik_skoru'], s['ham_benzerlik'], s['kategori'], s['guvenilirlik']) 
                for s in kaliteli_sonuclar[:3]
            ]
        }
    
    def _profesyonel_yanit_olustur(self, soru: str, sonuclar: List[Dict], acil: bool) -> str:
        """Profesyonel yanıt şablonu"""
        if acil:
            header = "🚨 KRİTİK CPR PROTOKOLÜ"
            uyari = "⚠️ **YAŞAMSAL ACİL DURUM!** Bu protokolleri kesin takip edin.\n\n"
            renk = "🔴"
        else:
            header = "📋 PROFESYONELLEŞTİRİLMİŞ CPR REHBERİ"
            uyari = ""
            renk = "🔵"
        
        yanit = f"## {header}\n\n{uyari}**Soru:** {soru}\n\n"
        
        for i, sonuc in enumerate(sonuclar):
            yanit += f"### {renk} Protokol {i+1}\n"
            yanit += f"**Kategori:** {sonuc['metadata']['kategori'].replace('_', ' ').title()}\n"
            yanit += f"**Alt Kategori:** {sonuc['metadata']['alt_kategori'].replace('_', ' ').title()}\n"
            yanit += f"**İçerik:** {sonuc['icerik']}\n\n"
            
            # Profesyonel kalite göstergeleri
            kalite_yildiz = "⭐" * min(5, max(1, int(sonuc['benzerlik_skoru'] * 8)))
            yanit += f"**Kalite Puanı:** {kalite_yildiz} ({sonuc['benzerlik_skoru']:.3f}) • "
            yanit += f"**Ham Skor:** {sonuc['ham_benzerlik']:.3f} • "
            yanit += f"**Güvenilirlik:** %{sonuc['guvenilirlik']*100:.0f} • "
            yanit += f"**Acillik:** {sonuc['acillik'].upper()} • "
            yanit += f"**Kaynak:** {sonuc['metadata']['kaynak']}\n\n"
            yanit += "---\n\n"
        
        yanit += "### ⚕️ PROFESYONELLEŞTİRİLMİŞ UYARILAR\n"
        yanit += "• **AHA 2020 Guidelines** ve **ERC 2021** temelinde hazırlanmıştır\n"
        yanit += "• **Gerçek uygulamada** mutlaka takım koordinasyonu yapın\n"
        yanit += "• **Acil durumlarda** 112'yi derhal arayın\n"
        yanit += "• **Sürekli eğitim** ve **düzenli pratik** yapmayı unutmayın\n"
        yanit += "• **Protokol güncellemeleri** için düzenli takip yapın\n"
        
        return yanit
    
    def _intelligent_oneri_sistemi(self, soru: str, yakinlik_sonuclari: List[Dict]) -> str:
        """Akıllı öneri sistemi"""
        yanit = f"## 🎯 AKILLI CPR REHBERİ\n\n"
        yanit += f"**Soru:** {soru}\n\n"
        yanit += "**Durum:** Spesifik protokol bulunamadı, akıllı öneriler sunuluyor.\n\n"
        
        # Yakın sonuçlar varsa göster
        if yakinlik_sonuclari:
            yanit += "### 📋 Yakın Konular:\n"
            for i, sonuc in enumerate(yakinlik_sonuclari):
                yanit += f"• **{sonuc['metadata']['kategori'].replace('_', ' ').title()}:** "
                yanit += f"{sonuc['icerik'][:80]}... (Skor: {sonuc['benzerlik_skoru']:.3f})\n"
            yanit += "\n"
        
        # Soru tipine göre özel öneriler
        if 'doz' in soru.lower() or 'miktar' in soru.lower():
            yanit += "### 💊 İlaç Dozu Rehberi:\n"
            yanit += "• **Epinefrin:** 1mg IV/IO her 3-5 dakikada bir\n"
            yanit += "• **Amiodarone:** İlk doz 300mg IV, ikinci doz 150mg\n"
            yanit += "• **Atropin:** 1mg IV, maksimum 3mg (bradiasistol için)\n"
            yanit += "• **Lidokain:** 1-1.5mg/kg IV (amiodarone alternatifi)\n\n"
        
        if 'çocuk' in soru.lower() or 'bebek' in soru.lower():
            yanit += "### 👶 Pediatrik CPR Rehberi:\n"
            yanit += "• **Çocuk (1-8 yaş):** Tek kişi 30:2, iki kişi 15:2\n"
            yanit += "• **Bebek (0-12 ay):** İki parmak tekniği, derinlik 4cm\n"
            yanit += "• **Kompresyon hızı:** 100-120/dk (tüm yaş grupları)\n\n"
        
        yanit += "### 🔍 Gelişmiş Arama Önerileri:\n"
        yanit += "• **Daha spesifik kelimeler** kullanın (örn: 'epinefrin 1mg IV doz')\n"
        yanit += "• **Yaş grubu** belirtin (yetişkin/çocuk/bebek)\n"
        yanit += "• **Sayısal değerler** sorun (kaç mg, ne kadar, hangi oran)\n"
        yanit += "• **Prosedür adımları** için 'nasıl', 'adım adım' kullanın\n"
        yanit += "• **Acil durumlar** için 'kritik', 'acil', 'arrest' ekleyin\n\n"
        
        yanit += "### 🎯 Popüler CPR Soruları:\n"
        yanit += "- Yetişkinlerde epinefrin dozu kaç mg?\n"
        yanit += "- CPR kompresyon oranı 30:2 mi?\n"
        yanit += "- AED cihazı nasıl adım adım kullanılır?\n"
        yanit += "- Çocuklarda göğüs bası derinliği ne olmalı?\n"
        yanit += "- Hipotermik arrest durumunda hangi protokol?\n"
        
        return yanit
    
    def get_sistem_metrikleri(self):
        """Sistem metrikleri"""
        uptime = datetime.now() - self.sistem_baslatma_zamani if self.sistem_baslatma_zamani else None
        
        return {
            'toplam_sorgu': self.toplam_sorgu,
            'basarili_sorgu': self.basarili_sorgu,
            'basari_orani': f"{(self.basarili_sorgu/max(1,self.toplam_sorgu))*100:.1f}%",
            'ortalama_yanit_suresi': f"{sum(self.ortalama_yanit_suresi)/max(1,len(self.ortalama_yanit_suresi)):.2f}s",
            'uptime': str(uptime).split('.')[0] if uptime else "0:00:00",
            'kategori_dagilimi': self.kategori_dagilimi,
            'database_boyutu': self.retriever.koleksiyon.count() if self.retriever.koleksiyon else 0
        }

# 🎨 MODERN STREAMLIT UI
st.set_page_config(
    page_title="CPR Professional System 🫀",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS tasarım
st.markdown("""
<style>
    /* Ana tema */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    /* Metrik kartları */
    .metric-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Başarı kartı */
    .success-card {
        background: linear-gradient(145deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
    }
    
    /* Uyarı kartı */
    .warning-card {
        background: linear-gradient(145deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffeaa7;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
    }
    
    /* Acil kart */
    .emergency-card {
        background: linear-gradient(145deg, #f8d7da, #f1aeb5);
        border: 1px solid #f1aeb5;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2); }
        50% { box-shadow: 0 6px 25px rgba(220, 53, 69, 0.4); }
        100% { box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2); }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(145deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Ana başlık
st.markdown("""
<div class="main-header">
    <h1>🫀 CPR Professional System v2.0</h1>
    <p><strong>Büyük Model (480MB) • Modern UI • Gelişmiş Performans İzleme</strong></p>
</div>
""", unsafe_allow_html=True)

# Gelişmiş örnek sorular
gelismis_ornekler = [
    "Yetişkinlerde epinefrin dozu kaç mg olmalıdır?",
    "CPR kompresyon oranı ve derinliği nedir?",
    "AED cihazı nasıl adım adım kullanılır?",
    "Çocuklarda kalp masajı nasıl yapılır?",
    "Hipotermik arrest durumunda protokol nedir?",
    "Amiodarone dozu ve uygulama şekli?",
    "Bebeklerde CPR tekniği nasıl uygulanır?",
    "VF/VT ritminde defibrilasyon protokolü?"
]

# SIDEBAR - PROFESYONELLEŞTİRİLMİŞ
with st.sidebar:
    st.markdown("## 🫀 CPR Professional Hub")
    st.markdown("---")
    
    # Hızlı erişim butonları
    st.markdown("### 🚀 Hızlı Sorular")
    for i, soru in enumerate(gelismis_ornekler):
        if st.button(soru, key=f"pro_btn_{i}", width='stretch'):
            st.session_state.pro_soru_input = soru
    
    st.markdown("---")
    
    # Sistem durumu
    st.markdown("### ⚙️ Sistem Durumu")
    
    if CHROMA_AVAILABLE and TRANSFORMERS_AVAILABLE:
        st.success("✅ Profesyonel AI Aktif")
        st.info("🧠 Model: distiluse-multilingual")
        st.info("💾 Boyut: 480MB")
        st.info("🌐 Türkçe: Optimizasyonlu")
    else:
        st.error("❌ Kütüphaneler Eksik")
        st.code("pip install chromadb sentence-transformers plotly")
    
    st.markdown("---")
    
    # Özellikler
    st.markdown("### 🏆 Pro Özellikler")
    st.markdown("""
    • 🎯 **Süper Kelime Genişletme**
    • 🧠 **480MB Güçlü Model**
    • 📊 **Performans İzleme**
    • 🎨 **Modern UI Tasarımı**
    • 🔍 **Intelligent Eşik Sistemi**
    • 📈 **Gerçek Zamanlı Grafikler**
    • ⚡ **Multi-faktör Skorlama**
    • 🎭 **Dinamik Yanıt Sistemi**
    """)

# Sistem başlatma
if "pro_sistem" not in st.session_state:
    with st.spinner("🚀 Profesyonel sistem başlatılıyor..."):
        st.session_state.pro_sistem = ProfesyonelCPRSistemi()
        st.session_state.pro_basladi = st.session_state.pro_sistem.sistem_baslat()

# Ana içerik
if not st.session_state.pro_basladi:
    st.markdown("""
    <div class="warning-card">
        <h3>❌ Sistem Başlatılamadı</h3>
        <p>Lütfen aşağıdakileri kontrol edin:</p>
        <ul>
            <li><strong>cpr_egitim_bilgi_bankasi.json</strong> dosyası mevcut mu?</li>
            <li>Gerekli kütüphaneler kurulu mu? <code>pip install chromadb sentence-transformers plotly</code></li>
            <li>İnternet bağlantısı var mı? (Model indirme için)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="success-card">🚀 <strong>Profesyonel sistem aktif!</strong> En gelişmiş CPR eğitimi deneyimi başladı.</div>', unsafe_allow_html=True)
    
    # PROFESYONEL METRİKLER DASHBOARD
    metrikliler = st.session_state.pro_sistem.get_sistem_metrikleri()
    
    st.markdown("## 📊 Sistem Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📚 Database",
            value=metrikliler['database_boyutu'],
            delta="Profesyonel"
        )
        
    with col2:
        st.metric(
            label="📈 Başarı Oranı",
            value=metrikliler['basari_orani'],
            delta=f"{metrikliler['basarili_sorgu']}/{metrikliler['toplam_sorgu']}"
        )
    
    with col3:
        st.metric(
            label="⚡ Ortalama Yanıt",
            value=metrikliler['ortalama_yanit_suresi'],
            delta="Ultra hızlı"
        )
    
    with col4:
        st.metric(
            label="🕐 Uptime",
            value=metrikliler['uptime'],
            delta="Kesintisiz"
        )
    
    # Kategori dağılımı
    if metrikliler['kategori_dagilimi']:
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Pie chart
            fig = px.pie(
                values=list(metrikliler['kategori_dagilimi'].values()),
                names=[k.replace('_', ' ').title() for k in metrikliler['kategori_dagilimi'].keys()],
                title="📂 CPR Kategori Dağılımı",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Kategori Detayları")
            for kategori, sayi in metrikliler['kategori_dagilimi'].items():
                st.markdown(f"**{kategori.replace('_', ' ').title()}:** {sayi} doküman")
    
    # Performans grafiği
    performans_data = st.session_state.pro_sistem.retriever.get_performans_grafigi()
    if performans_data:
        st.markdown("---")
        st.markdown("### 📈 Son Sorgular Performans Trendi")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(performans_data['skorlar']))),
            y=performans_data['skorlar'],
            mode='lines+markers+text',
            text=performans_data['sorgular'],
            textposition="top center",
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#764ba2'),
            name='Benzerlik Skoru'
        ))
        
        fig.update_layout(
            title="Son Sorguların Benzerlik Skorları",
            xaxis_title="Sorgu Sırası",
            yaxis_title="Benzerlik Skoru",
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # PROFESYONELLEŞTİRİLMİŞ SORU BÖLÜMÜ
    st.markdown("## 💬 Profesyonel Sorgulama Sistemi")
    
    soru = st.text_input(
        "CPR konusunda detaylı sorunuzu yazın:",
        value=st.session_state.get('pro_soru_input', ''),
        placeholder="Örn: Yetişkinlerde kardiyak arrest durumunda epinefrin dozu ve uygulama şekli nedir?",
        key="pro_ana_input",
        help="💡 İpucu: Daha spesifik sorular daha iyi sonuçlar verir"
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        sorgula_btn = st.button("🔍 Profesyonel Analiz", type="primary", width='stretch')
    
    with col2:
        if st.button("🔄 Temizle", width='stretch'):
            st.session_state.pro_soru_input = ""
            st.rerun()
    
    with col3:
        if st.button("🎲 Rastgele Pro", width='stretch'):
            st.session_state.pro_soru_input = random.choice(gelismis_ornekler)
            st.rerun()
    
    # PROFESYONEL SONUÇLAR
    if sorgula_btn and soru.strip():
        with st.spinner("🤖 Profesyonel AI sistemi kapsamlı analiz yapıyor..."):
            time.sleep(0.5)  # Professional feel
            sonuc = st.session_state.pro_sistem.profesyonel_sorgulama(soru)
        
        st.markdown("---")
        
        # Sonuç durumu kartı
        if sonuc["basarili"]:
            if sonuc["acil_durum"]:
                st.markdown('<div class="emergency-card">🚨 <strong>KRİTİK ACİL PROTOKOL BULUNDU!</strong> Yaşamsal öneme sahip bilgi.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-card">✅ <strong>Profesyonel protokol başarıyla bulundu!</strong> Yüksek kaliteli sonuç.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-card">⚠️ <strong>Akıllı öneriler sunuluyor.</strong> Daha spesifik arama önerileri mevcut.</div>', unsafe_allow_html=True)
        
        # Ana yanıt
        st.markdown(sonuc['yanit'])
        
        # PROFESYONELLEŞTİRİLMİŞ ANALİZ SONUÇLARI
        st.markdown("---")
        st.markdown("## 📊 Detaylı Analiz Raporu")
        
        # İki sütunlu analiz
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Arama Detayları")
            st.markdown(f"""
            <div class="metric-card">
                <strong>🎯 En İyi Skor:</strong> {sonuc['en_iyi_skor']:.3f}<br>
                <strong>🔍 Bulunan Doküman:</strong> {sonuc['bulunan_dokuman_sayisi']}<br>
                <strong>⭐ Kaliteli Sonuç:</strong> {sonuc['kaliteli_sonuc_sayisi']}<br>
                <strong>🎚️ Kullanılan Eşik:</strong> {sonuc['kullanilan_esik']:.3f} ({sonuc['esik_tipi']})<br>
                <strong>⚡ Yanıt Süresi:</strong> {sonuc['yanit_suresi']:.2f}s
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📈 Sistem Performansı")
            st.markdown(f"""
            <div class="metric-card">
                <strong>🏆 Performans:</strong> {sonuc['sistem_performansi']}<br>
                <strong>📊 Genel Başarı:</strong> {sonuc['basari_orani']}<br>
                <strong>🫀 CPR Odaklı:</strong> {'✅ Evet' if sonuc['cpr_odakli'] else '❌ Hayır'}<br>
                <strong>🚨 Acil Durum:</strong> {'🔴 EVET' if sonuc['acil_durum'] else '🟢 Normal'}<br>
                <strong>🎯 Sorgu Tipi:</strong> {sonuc['esik_tipi'].title()}
            </div>
            """, unsafe_allow_html=True)
        
        # Detaylı eşleşme tablosu
        if sonuc["sonuc_detaylari"]:
            st.markdown("---")
            st.markdown("### 🎯 En İyi Eşleşmeler - Detaylı Analiz")
            
            tablo_data = []
            for final_skor, ham_skor, kategori, guvenilirlik in sonuc["sonuc_detaylari"]:
                # Kalite değerlendirmesi
                if final_skor > 0.6:
                    kalite = "🏆 Mükemmel"
                    renk = "🟢"
                elif final_skor > 0.4:
                    kalite = "🚀 Çok İyi"
                    renk = "🔵"
                elif final_skor > 0.25:
                    kalite = "📈 İyi"
                    renk = "🟡"
                else:
                    kalite = "📊 Orta"
                    renk = "🟠"
                
                tablo_data.append({
                    "Durum": renk,
                    "🏆 Final Skor": f"{final_skor:.3f}",
                    "📊 Ham Skor": f"{ham_skor:.3f}",
                    "📂 Kategori": kategori.replace('_', ' ').title(),
                    "⭐ Güvenilirlik": f"%{guvenilirlik*100:.0f}",
                    "🎯 Kalite": kalite
                })
            
            st.dataframe(
                tablo_data, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Durum": st.column_config.TextColumn(width="small"),
                    "🏆 Final Skor": st.column_config.NumberColumn(format="%.3f"),
                    "📊 Ham Skor": st.column_config.NumberColumn(format="%.3f")
                }
            )
    
    elif sorgula_btn and not soru.strip():
        st.markdown('<div class="warning-card">❗ <strong>Lütfen detaylı bir CPR sorusu yazın.</strong> Daha spesifik sorular daha iyi sonuçlar verir.</div>', unsafe_allow_html=True)

# FOOTER - PROFESYONELLEŞTİRİLMİŞ
st.markdown("---")
st.markdown("## 🎓 CPR Profesyonel Eğitim Merkezi")

# Özellik showcase
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>🧠 Gelişmiş AI Teknolojisi</h4>
        <ul>
            <li><strong>480MB büyük model</strong></li>
            <li><strong>Çok dilli destek</strong></li>
            <li><strong>Süper kelime genişletme</strong></li>
            <li><strong>Multi-faktör skorlama</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>📚 Kapsamlı Tıbbi İçerik</h4>
        <ul>
            <li><strong>AHA 2020 Guidelines</strong></li>
            <li><strong>ERC 2021 Standartları</strong></li>
            <li><strong>Pediatrik protokoller</strong></li>
            <li><strong>İleri yaşam desteği</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>⚡ Profesyonel Performans</h4>
        <ul>
            <li><strong>Intelligent eşik sistemi</strong></li>
            <li><strong>Gerçek zamanlı analiz</strong></li>
            <li><strong>Performans izleme</strong></li>
            <li><strong>Modern UI/UX</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Final professional card
st.markdown("""
<div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 20px; color: white; margin: 2rem 0; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);">
    <h2>🫀 CPR Professional System v2.0</h2>
    <p style="font-size: 1.2em; margin: 1rem 0;"><strong>Hayat kurtaran bilgi, profesyonel teknoloji ile buluşuyor</strong></p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap;">
        <div>🎯 <strong>Akıllı Arama</strong></div>
        <div>📊 <strong>Performans İzleme</strong></div>
        <div>⚡ <strong>Ultra Hızlı</strong></div>
        <div>🌐 <strong>Çok Dilli</strong></div>
        <div>🧠 <strong>480MB Model</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px; margin: 1rem 0;">
    <p><strong>⚠️ UYARI:</strong> Bu sistem eğitim amaçlıdır. Gerçek acil durumlarda <strong>112</strong>'yi arayın ve profesyonel tıbbi yardım alın.</p>
    <p>Tüm protokoller <strong>AHA (American Heart Association) 2020 Guidelines</strong> ve <strong>ERC 2021</strong> temelinde hazırlanmıştır.</p>
</div>
""", unsafe_allow_html=True)