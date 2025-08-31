# 🚀 QUERY CHUNKING + UI FIX - CPR Professional System v2.1
# Bu kodu cpr_chunking_enhanced.py olarak kaydet

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
import re

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

class QueryChunker:
    """Akıllı sorgu parçalama sistemi"""
    
    def __init__(self):
        # Chunking kuralları
        self.ayirici_kelimeler = [
            've', 'ile', 'ayrıca', 'hem', 'ek olarak', 'bunun yanında',
            'and', 'with', 'also', 'plus', 'additionally'
        ]
        
        self.soru_kelimeleri = [
            'nasıl', 'nedir', 'neden', 'ne zaman', 'nerede', 'kaç', 'hangi',
            'how', 'what', 'why', 'when', 'where', 'which', 'how many'
        ]
        
        self.cpr_ana_konular = [
            'cpr', 'kompresyon', 'solunum', 'aed', 'defibrilasyon',
            'epinefrin', 'amiodarone', 'atropin', 'hipotermik',
            'yetişkin', 'çocuk', 'bebek', 'kalp', 'nabız'
        ]
    
    def akilli_chunking(self, sorgu: str) -> List[str]:
        """Akıllı sorgu parçalama"""
        if len(sorgu.split()) <= 8:
            return [sorgu]  # Kısa sorular için chunking yapma
        
        print(f"[CHUNKING] Uzun sorgu tespit edildi: '{sorgu[:50]}...'")
        
        chunks = []
        
        # 1. Virgül ve nokta ile böl
        ana_parcalar = re.split(r'[,.]', sorgu)
        
        # 2. Her ana parçayı işle
        for parca in ana_parcalar:
            parca = parca.strip()
            if not parca:
                continue
                
            # 3. Ayırıcı kelimelerle böl
            for ayirici in self.ayirici_kelimeler:
                if ayirici in parca.lower():
                    alt_parcalar = parca.split(ayirici)
                    for alt_parca in alt_parcalar:
                        alt_parca = alt_parca.strip()
                        if len(alt_parca.split()) >= 3:  # Minimum 3 kelime
                            chunks.append(alt_parca)
                    break
            else:
                # Ayırıcı yoksa direkt ekle
                if len(parca.split()) >= 3:
                    chunks.append(parca)
        
        # 4. Soru kelimesi olan parçaları önceliklendir
        soru_chunks = []
        normal_chunks = []
        
        for chunk in chunks:
            if any(soru in chunk.lower() for soru in self.soru_kelimeleri):
                soru_chunks.append(chunk)
            else:
                normal_chunks.append(chunk)
        
        # 5. CPR kelimesi ekle (eğer yoksa)
        enhanced_chunks = []
        for chunk in soru_chunks + normal_chunks:
            if not any(cpr_kelime in chunk.lower() for cpr_kelime in self.cpr_ana_konular):
                # CPR konteksti ekle
                chunk = f"CPR {chunk}"
            enhanced_chunks.append(chunk)
        
        # 6. Orijinal sorguyu da ekle (son şans)
        enhanced_chunks.append(sorgu)
        
        print(f"[CHUNKING] {len(enhanced_chunks)} chunk oluşturuldu:")
        for i, chunk in enumerate(enhanced_chunks):
            print(f"  Chunk {i+1}: '{chunk[:40]}...'")
        
        return enhanced_chunks[:5]  # Maksimum 5 chunk

class EnhancedRetriever:
    """Query Chunking ile geliştirilmiş retriever"""
    
    def __init__(self):
        self.koleksiyon_adi = "cpr_chunking_v1"
        self.chroma_client = None
        self.koleksiyon = None
        self.model = None
        self.chunker = QueryChunker()
        self.performans_gecmisi = []
        
        # Gelişmiş kelime haritası (öncekinden daha kapsamlı)
        self.mega_kelime_haritasi = {
            # İlaç terimleri
            'epinefrin': ['epinephrine', 'adrenaline', 'adrenalin', 'vazopresor', 'vazopressör', 'noradrenalin'],
            'amiodarone': ['amiodaron', 'antiaritmik', 'kardiak ilaç', 'ritim düzenleyici', 'kordarone'],
            'atropin': ['atropine', 'antikolinerjik', 'bradikardi ilacı', 'atropine sulfat'],
            'lidokain': ['lidocaine', 'lokal anestezik', 'antiaritmik', 'xylocaine'],
            'magnezyum': ['magnesium', 'mg', 'elektrolit', 'magnesium sulfat'],
            
            # CPR terimleri - KAPSAMLI
            'cpr': ['cardiopulmonary resuscitation', 'kalp masajı', 'canlandırma', 'resüsitasyon', 
                   'yaşam desteği', 'temel yaşam desteği', 'ileri yaşam desteği', 'kardiyopulmoner',
                   'kalp akciğer canlandırma', 'temel canlandırma'],
            'kompresyon': ['compression', 'basınç', 'göğüs basısı', 'masaj', 'basma', 'göğüs kompresyonu'],
            'ventilasyon': ['solunum', 'nefes', 'hava verme', 'breathing', 'oksijen', 'yapay solunum',
                           'mouth to mouth', 'ağızdan ağıza', 'bag mask'],
            'defibrilasyon': ['defibrillation', 'şok', 'elektrik şoku', 'kardiyoversiyon', 'DC şok'],
            
            # Cihaz terimleri  
            'aed': ['automated external defibrillator', 'defibrillatör', 'şok cihazı', 'otomatik defibrillatör',
                   'external defibrillator', 'aed cihazı'],
            'monitör': ['monitor', 'ekg', 'ecg', 'kalp ritmi takibi', 'cardiac monitor'],
            
            # Anatomi terimleri - GENİŞ
            'kalp': ['heart', 'kardiak', 'miyokard', 'ventrikül', 'atriyum', 'cardiac', 'jantung'],
            'göğüs': ['chest', 'toraks', 'göğüs kafesi', 'sternum', 'thorax'],
            'nabız': ['pulse', 'kalp ritmi', 'heartbeat', 'nabız hızı', 'heart rate'],
            
            # Yaş grupları - DETAYLI
            'yetişkin': ['adult', 'erişkin', 'büyük', '18 yaş üzeri', 'mature', 'grown up'],
            'çocuk': ['child', 'pediatrik', 'küçük', '1-8 yaş', 'okul çağı', 'pediatric', 'kid'],
            'bebek': ['infant', 'baby', 'yenidoğan', '0-12 ay', 'süt çocuğu', 'newborn'],
            
            # Ölçü birimleri - KAPSAMLI
            'doz': ['dose', 'miktar', 'amount', 'dozaj', 'dosis', 'quantity'],
            'mg': ['milligram', 'miligram', 'ml', 'cc', 'mcg', 'microgram'],
            'oran': ['ratio', 'rate', 'frequency', 'hız', 'frekans', 'proportion'],
            'derinlik': ['depth', 'derinlik', 'cm', 'santimetre', 'centimeter'],
            
            # Acil durum terimleri - GENİŞ
            'arrest': ['kalp durması', 'kardiyak arrest', 'cardiac arrest', 'ani ölüm', 'sudden death'],
            'hipotermik': ['hypothermic', 'soğuk', 'düşük sıcaklık', 'hipotermi', 'hypothermia'],
            'vf': ['ventricular fibrillation', 'ventrikül fibrilasyonu', 'v-fib'],
            'vt': ['ventricular tachycardia', 'ventrikül taşikardisi', 'v-tach'],
            'asistol': ['asystole', 'düz çizgi', 'kalp durması', 'flatline'],
            
            # Soru kelimeleri - KAPSAMLI
            'nasıl': ['how', 'ne şekilde', 'hangi yöntem', 'prosedür', 'adımlar'],
            'nedir': ['what', 'ne', 'tanım', 'definition'],
            'kaç': ['how much', 'how many', 'ne kadar', 'miktar', 'quantity'],
            'nerede': ['where', 'hangi bölge', 'lokasyon', 'location'],
            'ne zaman': ['when', 'hangi durumda', 'zamanı', 'timing'],
            
            # Yeni eklenen - ÖZEL DURUMLAR
            'adım': ['step', 'adımlar', 'procedure', 'protocol', 'stages'],
            'dikkat': ['attention', 'care', 'caution', 'warning', 'notice'],
            'risk': ['risk', 'danger', 'hazard', 'complication', 'side effect'],
            'enfeksiyon': ['infection', 'contamination', 'disease', 'pathogen']
        }
    
    def sistem_baslat(self):
        """Gelişmiş sistem başlatma"""
        if not CHROMA_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            st.error("❌ pip install chromadb sentence-transformers plotly")
            return False
        
        try:
            with st.spinner("🚀 Query Chunking sistemi başlatılıyor..."):
                # ChromaDB
                self.chroma_client = chromadb.Client(Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                ))
                
                # Güçlü model
                st.info("📥 Güçlü Türkçe modeli yükleniyor...")
                self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased')
                
                try:
                    self.koleksiyon = self.chroma_client.get_collection(self.koleksiyon_adi)
                    st.success(f"✅ Chunking database hazır: {self.koleksiyon.count()} doküman")
                except:
                    self.koleksiyon = self.chroma_client.create_collection(
                        name=self.koleksiyon_adi,
                        metadata={"version": "chunking_v1", "model": "distiluse-multilingual"}
                    )
                    st.info("🆕 Yeni chunking database oluşturuluyor...")
                
            return True
            
        except Exception as e:
            st.error(f"❌ Sistem hatası: {str(e)}")
            return False
    
    def _mega_kelime_genisletme(self, metin: str) -> str:
        """Mega kelime genişletme sistemi"""
        genisletilmis = metin.lower()
        
        # Her kelime için kapsamlı eşleştirme
        for anahtar, esanlamlilar in self.mega_kelime_haritasi.items():
            # Ana kelime varsa eşanlamlıları ekle
            if anahtar in genisletilmis:
                genisletilmis += " " + " ".join(esanlamlilar)
            
            # Eşanlamlılardan biri varsa ana kelimeyi ekle
            for esanlamli in esanlamlilar:
                if esanlamli.lower() in genisletilmis.lower() and anahtar not in genisletilmis:
                    genisletilmis += " " + anahtar
        
        # Özel durum işlemleri - GENİŞ
        if any(kelime in genisletilmis for kelime in ["kaç", "ne kadar", "miktar"]):
            genisletilmis += " doz miktar mg amount dose quantity"
        
        if any(kelime in genisletilmis for kelime in ["nasıl", "adım", "prosedür"]):
            genisletilmis += " prosedür adım yöntem protokol procedure steps"
            
        if any(kelime in genisletilmis for kelime in ["dikkat", "risk", "önlem"]):
            genisletilmis += " dikkat edilmesi gereken risk önlem caution warning"
            
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
                metadata={"version": "chunking_v1", "model": "distiluse-multilingual"}
            )
        
        progress = st.progress(0)
        status = st.empty()
        time_start = time.time()
        
        ids, embeddings, metadatas, documents = [], [], [], []
        
        for i, dok in enumerate(dokumanlar):
            ids.append(dok.get('id', str(uuid.uuid4())))
            
            # MEGA genişletilmiş içerik
            temel_icerik = dok['icerik']
            kategori = dok.get('kategori', '')
            alt_kategori = dok.get('alt_kategori', '')
            
            # Tam kapsamlı genişletme
            tam_icerik = f"{temel_icerik} {kategori} {alt_kategori}"
            genisletilmis_icerik = self._mega_kelime_genisletme(tam_icerik)
            
            # Güçlü embedding
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
                'kelime_sayisi': len(temel_icerik.split()),
                'chunking_ready': True
            })
            
            documents.append(temel_icerik)
            
            # Progress
            elapsed = time.time() - time_start
            if i > 0:
                eta = (elapsed / i) * (len(dokumanlar) - i)
                status.text(f"📊 Chunking hazırlık: {i + 1}/{len(dokumanlar)} • ETA: {eta:.1f}s")
            
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
            status.success(f"✅ {len(dokumanlar)} chunking-ready doküman eklendi! ({total_time:.1f}s)")
            time.sleep(1.5)
            progress.empty()
            status.empty()
            return True
            
        except Exception as e:
            st.error(f"❌ Ekleme hatası: {str(e)}")
            return False
    
    def chunked_arama(self, sorgu: str, top_k: int = 10) -> List[Dict]:
        """🚀 QUERY CHUNKING ile gelişmiş arama"""
        if not self.koleksiyon or not self.model:
            return []
        
        try:
            # 1. SORGUYU CHUNKLA
            chunks = self.chunker.akilli_chunking(sorgu)
            
            print(f"[CHUNKED SEARCH] '{sorgu[:30]}...' için {len(chunks)} chunk ile arama")
            
            # 2. HER CHUNK İÇİN ARAMA YAP
            tum_sonuclar = {}  # ID bazında sonuçları topla
            chunk_skorlari = {}  # Her chunk için ayrı skorlar
            
            for chunk_idx, chunk in enumerate(chunks):
                # Chunk'ı genişlet
                genisletilmis_chunk = self._mega_kelime_genisletme(chunk)
                
                # Chunk embedding
                chunk_embedding = self.model.encode(genisletilmis_chunk).tolist()
                
                # Arama yap
                chunk_sonuclar = self.koleksiyon.query(
                    query_embeddings=[chunk_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Sonuçları işle
                if chunk_sonuclar['documents'] and len(chunk_sonuclar['documents'][0]) > 0:
                    for i in range(len(chunk_sonuclar['documents'][0])):
                        doc_id = chunk_sonuclar['ids'][0][i]
                        distance = chunk_sonuclar['distances'][0][i]
                        similarity = max(0.0, 1.0 - distance)
                        
                        # Chunk ağırlığı (ilk chunk'lar daha önemli)
                        chunk_weight = 1.0 - (chunk_idx * 0.1)
                        weighted_score = similarity * chunk_weight
                        
                        if doc_id not in tum_sonuclar:
                            tum_sonuclar[doc_id] = {
                                'id': doc_id,
                                'icerik': chunk_sonuclar['documents'][0][i],
                                'metadata': chunk_sonuclar['metadatas'][0][i],
                                'chunk_skorlari': [],
                                'max_skor': 0,
                                'ortalama_skor': 0,
                                'chunk_sayisi': 0
                            }
                        
                        # Chunk skorlarını topla
                        tum_sonuclar[doc_id]['chunk_skorlari'].append({
                            'chunk_idx': chunk_idx,
                            'chunk_text': chunk[:30] + '...',
                            'similarity': similarity,
                            'weighted_score': weighted_score
                        })
                        
                        # Max ve ortalama hesapla
                        skorlar = [cs['weighted_score'] for cs in tum_sonuclar[doc_id]['chunk_skorlari']]
                        tum_sonuclar[doc_id]['max_skor'] = max(skorlar)
                        tum_sonuclar[doc_id]['ortalama_skor'] = sum(skorlar) / len(skorlar)
                        tum_sonuclar[doc_id]['chunk_sayisi'] = len(skorlar)
            
            # 3. SONUÇLARI BİRLEŞTİR VE SKORLA
            final_sonuclar = []
            for doc_id, data in tum_sonuclar.items():
                # CHUNKING BONUSU: Birden fazla chunk'ta bulunan dokümanlar bonus alır
                multi_chunk_bonus = 1.0 + (data['chunk_sayisi'] - 1) * 0.15
                
                # Güvenilirlik bonusu
                guvenilirlik = data['metadata'].get('guvenilirlik', 0.8)
                acillik = data['metadata'].get('acillik_seviyesi', 'normal')
                acillik_bonus = 1.2 if acillik == 'kritik' else 1.0
                
                # FINAL SKOR = (Max + Ortalama) / 2 * Bonuslar
                final_score = ((data['max_skor'] + data['ortalama_skor']) / 2) * multi_chunk_bonus * (0.7 + 0.3 * guvenilirlik) * acillik_bonus
                
                final_sonuclar.append({
                    'id': doc_id,
                    'icerik': data['icerik'],
                    'benzerlik_skoru': final_score,
                    'ham_max_skor': data['max_skor'],
                    'ham_ortalama_skor': data['ortalama_skor'],
                    'chunk_sayisi': data['chunk_sayisi'],
                    'chunk_detaylari': data['chunk_skorlari'],
                    'metadata': data['metadata'],
                    'kategori': data['metadata'].get('kategori', 'genel'),
                    'guvenilirlik': guvenilirlik,
                    'acillik': acillik,
                    'multi_chunk_bonus': multi_chunk_bonus
                })
            
            # 4. SKORLARA GÖRE SIRALA
            final_sonuclar.sort(key=lambda x: x['benzerlik_skoru'], reverse=True)
            
            # 5. PERFORMANS KAYDI
            if final_sonuclar:
                self.performans_gecmisi.append({
                    'timestamp': datetime.now(),
                    'sorgu': sorgu,
                    'chunk_sayisi': len(chunks),
                    'en_iyi_skor': final_sonuclar[0]['benzerlik_skoru'],
                    'sonuc_sayisi': len(final_sonuclar),
                    'chunking_bonus_kullanildi': True
                })
                
                if len(self.performans_gecmisi) > 20:
                    self.performans_gecmisi.pop(0)
            
            # 6. DEBUG BİLGİSİ
            print(f"[CHUNKED RESULTS] '{sorgu[:30]}...' için {len(final_sonuclar)} birleştirilmiş sonuç:")
            for i, sonuc in enumerate(final_sonuclar[:3]):
                print(f"  {i+1}. Final: {sonuc['benzerlik_skoru']:.3f}, Max: {sonuc['ham_max_skor']:.3f}, "
                      f"Chunks: {sonuc['chunk_sayisi']}, Kategori: {sonuc['kategori']}")
            
            return final_sonuclar
            
        except Exception as e:
            st.error(f"❌ Chunked arama hatası: {str(e)}")
            return []
    
    def get_performans_grafigi(self):
        """Chunking performans grafik verisi"""
        if not self.performans_gecmisi:
            return None
        
        return {
            'timestamps': [p['timestamp'] for p in self.performans_gecmisi],
            'skorlar': [p['en_iyi_skor'] for p in self.performans_gecmisi],
            'chunk_sayilari': [p['chunk_sayisi'] for p in self.performans_gecmisi],
            'sorgular': [p['sorgu'][:15] + '...' if len(p['sorgu']) > 15 else p['sorgu'] 
                        for p in self.performans_gecmisi]
        }

class ChunkingCPRSistemi:
    """Query Chunking ile geliştirilmiş CPR sistemi"""
    
    def __init__(self):
        self.retriever = EnhancedRetriever()
        self.bilgi_bankasi = []
        self.toplam_sorgu = 0
        self.basarili_sorgu = 0
        self.chunking_kullanim_sayisi = 0
        self.sistem_baslatma_zamani = None
        
        # Performans metrikleri
        self.ortalama_yanit_suresi = []
        self.kategori_dagilimi = {}
        self.chunk_istatistikleri = {
            'toplam_chunk': 0,
            'basarili_chunk': 0,
            'ortalama_chunk_per_sorgu': 0
        }
    
    def sistem_baslat(self):
        """Chunking sistemi başlatma"""
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
                st.info("🔄 Chunking database oluşturuluyor...")
                if not self.retriever.dokumanlar_ekle(self.bilgi_bankasi, temizle=True):
                    return False
            
            return True
            
        except Exception as e:
            st.error(f"❌ Sistem hatası: {str(e)}")
            return False
    
    def chunking_sorgulama(self, soru: str) -> Dict:
        """🚀 CHUNKING ile gelişmiş sorgulama"""
        start_time = time.time()
        self.toplam_sorgu += 1
        
        # Chunking kullanım kararı
        kelime_sayisi = len(soru.split())
        chunking_kullan = kelime_sayisi > 8
        
        if chunking_kullan:
            self.chunking_kullanim_sayisi += 1
            print(f"[CHUNKING ACTIVE] {kelime_sayisi} kelimelik sorgu için chunking aktif")
            
            # Chunked arama
            sonuclar = self.retriever.chunked_arama(soru, top_k=8)
        else:
            print(f"[NORMAL SEARCH] {kelime_sayisi} kelimelik sorgu için normal arama")
            # Normal arama (chunking olmadan)
            genisletilmis_sorgu = self.retriever._mega_kelime_genisletme(soru)
            sorgu_embedding = self.retriever.model.encode(genisletilmis_sorgu).tolist()
            
            normal_sonuclar = self.retriever.koleksiyon.query(
                query_embeddings=[sorgu_embedding],
                n_results=8,
                include=["documents", "metadatas", "distances"]
            )
            
            # Normal sonuçları formatla
            sonuclar = []
            if normal_sonuclar['documents'] and len(normal_sonuclar['documents'][0]) > 0:
                for i in range(len(normal_sonuclar['documents'][0])):
                    distance = normal_sonuclar['distances'][0][i]
                    similarity = max(0.0, 1.0 - distance)
                    
                    sonuclar.append({
                        'id': normal_sonuclar['ids'][0][i],
                        'icerik': normal_sonuclar['documents'][0][i],
                        'benzerlik_skoru': similarity,
                        'ham_max_skor': similarity,
                        'ham_ortalama_skor': similarity,
                        'chunk_sayisi': 1,
                        'metadata': normal_sonuclar['metadatas'][0][i],
                        'kategori': normal_sonuclar['metadatas'][0][i].get('kategori', 'genel'),
                        'guvenilirlik': normal_sonuclar['metadatas'][0][i].get('guvenilirlik', 0.8)
                    })
        
        # Intelligent eşik sistemi (chunking'e göre ayarlanmış)
        esik_kurallari = {
            'doz_miktar': (['doz', 'miktar', 'mg', 'kaç', 'ne kadar'], 0.03 if chunking_kullan else 0.05),
            'acil_kritik': (['acil', 'kritik', 'emergency', 'arrest', 'durma'], 0.02 if chunking_kullan else 0.04),
            'prosedur': (['nasıl', 'how', 'adım', 'yöntem', 'prosedür'], 0.04 if chunking_kullan else 0.06),
            'tanım': (['nedir', 'what', 'tanım', 'ne'], 0.06 if chunking_kullan else 0.08),
            'genel': ([], 0.08 if chunking_kullan else 0.12)
        }
        
        # Uygun eşiği bul
        kullanilan_esik = esik_kurallari['genel'][1]
        esik_tipi = 'genel'
        
        for tip, (kelimeler, esik_degeri) in esik_kurallari.items():
            if any(kelime in soru.lower() for kelime in kelimeler):
                kullanilan_esik = esik_degeri
                esik_tipi = tip
                break
        
        kaliteli_sonuclar = [s for s in sonuclar if s['benzerlik_skoru'] > kullanilan_esik]
        
        # CPR ve acil durum analizi
        cpr_kelimeler = ['cpr', 'kalp', 'resüsitasyon', 'defibrilasyon', 'epinefrin', 'aed', 'kompresyon']
        cpr_odakli = any(kelime in soru.lower() for kelime in cpr_kelimeler)
        
        acil_kelimeler = ['acil', 'kritik', 'emergency', 'arrest', 'durma', 'kriz']
        acil_durum = any(kelime in soru.lower() for kelime in acil_kelimeler)
        
        # Yanıt oluşturma
        if len(kaliteli_sonuclar) >= 1:
            self.basarili_sorgu += 1
            yanit = self._chunking_yanit_olustur(soru, kaliteli_sonuclar[:3], acil_durum, chunking_kullan)
            basarili = True
        else:
            yanit = self._chunking_oneri_sistemi(soru, sonuclar[:2], chunking_kullan)
            basarili = False
        
        # Performans hesaplama
        yanit_suresi = time.time() - start_time
        self.ortalama_yanit_suresi.append(yanit_suresi)
        
        if len(self.ortalama_yanit_suresi) > 10:
            self.ortalama_yanit_suresi.pop(0)
        
        # Chunk istatistikleri güncelle
        if chunking_kullan and kaliteli_sonuclar:
            self.chunk_istatistikleri['toplam_chunk'] += sum(s.get('chunk_sayisi', 1) for s in kaliteli_sonuclar[:3])
            self.chunk_istatistikleri['basarili_chunk'] += len(kaliteli_sonuclar)
            self.chunk_istatistikleri['ortalama_chunk_per_sorgu'] = self.chunk_istatistikleri['toplam_chunk'] / max(1, self.chunking_kullanim_sayisi)
        
        # Skor değerlendirmesi
        if kaliteli_sonuclar:
            en_iyi_skor = kaliteli_sonuclar[0]['benzerlik_skoru']
            if en_iyi_skor > 0.8:
                performans = "🏆 Mükemmel"
            elif en_iyi_skor > 0.6:
                performans = "🚀 Çok İyi"
            elif en_iyi_skor > 0.4:
                performans = "📈 İyi"
            elif en_iyi_skor > 0.2:
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
            "chunking_kullanildi": chunking_kullan,
            "chunking_istatistikleri": self.chunk_istatistikleri.copy(),
            "basari_orani": f"{(self.basarili_sorgu/max(1,self.toplam_sorgu))*100:.1f}%",
            "sonuc_detaylari": [
                (s['benzerlik_skoru'], s.get('ham_max_skor', 0), s.get('chunk_sayisi', 1), s['kategori'], s['guvenilirlik']) 
                for s in kaliteli_sonuclar[:3]
            ]
        }
    
    def _chunking_yanit_olustur(self, soru: str, sonuclar: List[Dict], acil: bool, chunking_kullan: bool) -> str:
        """Chunking-aware yanıt şablonu"""
        if acil:
            header = "🚨 KRİTİK CPR PROTOKOLÜ"
            uyari = "⚠️ **YAŞAMSAL ACİL DURUM!** Bu protokolleri kesin takip edin.\n\n"
            renk = "🔴"
        else:
            header = "📋 CHUNKING-ENHANCEDCPRREHBERİ" if chunking_kullan else "📋 CPR REHBERİ"
            uyari = ""
            renk = "🔵"
        
        yanit = f"## {header}\n\n{uyari}**Soru:** {soru}\n\n"
        
        if chunking_kullan:
            yanit += "🧩 **Query Chunking Aktif:** Uzun sorgunuz parçalara bölünerek analiz edildi.\n\n"
        
        for i, sonuc in enumerate(sonuclar):
            yanit += f"### {renk} Protokol {i+1}\n"
            yanit += f"**Kategori:** {sonuc['metadata']['kategori'].replace('_', ' ').title()}\n"
            yanit += f"**Alt Kategori:** {sonuc['metadata']['alt_kategori'].replace('_', ' ').title()}\n"
            yanit += f"**İçerik:** {sonuc['icerik']}\n\n"
            
            # Chunking detayları
            if chunking_kullan and 'chunk_detaylari' in sonuc:
                yanit += f"**🧩 Chunk Analizi:** {sonuc['chunk_sayisi']} parça, "
                if sonuc.get('multi_chunk_bonus', 1.0) > 1.0:
                    yanit += f"Multi-chunk bonus: +{((sonuc['multi_chunk_bonus']-1)*100):.0f}%"
                yanit += "\n"
            
            # Kalite göstergeleri
            kalite_yildiz = "⭐" * min(5, max(1, int(sonuc['benzerlik_skoru'] * 6)))
            yanit += f"**Kalite Puanı:** {kalite_yildiz} ({sonuc['benzerlik_skoru']:.3f}) • "
            if chunking_kullan:
                yanit += f"**Max Chunk:** {sonuc.get('ham_max_skor', 0):.3f} • "
                yanit += f"**Avg Chunk:** {sonuc.get('ham_ortalama_skor', 0):.3f} • "
            yanit += f"**Güvenilirlik:** %{sonuc['guvenilirlik']*100:.0f} • "
            yanit += f"**Kaynak:** {sonuc['metadata']['kaynak']}\n\n"
            yanit += "---\n\n"
        
        yanit += "### ⚕️ PROFESYONELLEŞTİRİLMİŞ UYARILAR\n"
        yanit += "• **AHA 2020 Guidelines** ve **ERC 2021** temelinde hazırlanmıştır\n"
        yanit += "• **Gerçek uygulamada** mutlaka takım koordinasyonu yapın\n"
        yanit += "• **Acil durumlarda** 112'yi derhal arayın\n"
        if chunking_kullan:
            yanit += "• **Query Chunking** teknolojisi ile geliştirilmiş analiz\n"
        yanit += "• **Sürekli eğitim** ve **düzenli pratik** yapmayı unutmayın\n"
        
        return yanit
    
    def _chunking_oneri_sistemi(self, soru: str, yakin_sonuclar: List[Dict], chunking_kullan: bool) -> str:
        """Chunking-aware akıllı öneri sistemi"""
        yanit = f"## 🎯 AKILLI CPR REHBERİ\n\n"
        yanit += f"**Soru:** {soru}\n\n"
        yanit += "**Durum:** Spesifik protokol bulunamadı"
        
        if chunking_kullan:
            yanit += ", query chunking kullanıldı ancak yeterli eşleşme yok.\n\n"
        else:
            yanit += ".\n\n"
        
        # Yakın sonuçlar varsa göster
        if yakin_sonuclar:
            yanit += "### 📋 Yakın Konular:\n"
            for i, sonuc in enumerate(yakin_sonuclar):
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
        
        if any(kelime in soru.lower() for kelime in ['nasıl', 'adım', 'prosedür']):
            yanit += "### 📋 Prosedür Rehberi:\n"
            yanit += "• **CPR Adımları:** Yanıtsızlık kontrolü → Nabız kontrolü → 30:2 → AED\n"
            yanit += "• **AED Kullanımı:** Aç → Elektrot yerleştir → Analiz → Şok (gerekirse)\n"
            yanit += "• **Hava Yolu:** Head-tilt chin-lift → Bag-mask ventilasyon\n\n"
        
        yanit += "### 🔍 Gelişmiş Arama Önerileri:\n"
        
        if not chunking_kullan:
            yanit += "• **Daha detaylı sorular** sorun (sistem otomatik chunking yapacak)\n"
        else:
            yanit += "• **Farklı kelimelerle** tekrar deneyin\n"
        
        yanit += "• **Spesifik terimler** kullanın (epinefrin, AED, kompresyon)\n"
        yanit += "• **Yaş grubu** belirtin (yetişkin/çocuk/bebek)\n"
        yanit += "• **Sayısal değerler** sorun (kaç mg, ne kadar, hangi oran)\n"
        yanit += "• **CPR kelimesi** ekleyin sorguya\n\n"
        
        yanit += "### 🎯 Popüler CPR Soruları:\n"
        yanit += "- Yetişkinlerde epinefrin dozu ve uygulama şekli nedir?\n"
        yanit += "- CPR kompresyon oranı ve derinliği nasıl olmalı?\n"
        yanit += "- AED cihazı nasıl adım adım kullanılır?\n"
        yanit += "- Çocuklarda kalp masajı nasıl uygulanır?\n"
        yanit += "- Hipotermik arrest durumunda hangi protokol uygulanır?\n"
        
        return yanit
    
    def get_sistem_metrikleri(self):
        """Chunking-enhanced sistem metrikleri"""
        uptime = datetime.now() - self.sistem_baslatma_zamani if self.sistem_baslatma_zamani else None
        
        chunking_oran = f"{(self.chunking_kullanim_sayisi/max(1,self.toplam_sorgu))*100:.1f}%"
        
        return {
            'toplam_sorgu': self.toplam_sorgu,
            'basarili_sorgu': self.basarili_sorgu,
            'basari_orani': f"{(self.basarili_sorgu/max(1,self.toplam_sorgu))*100:.1f}%",
            'ortalama_yanit_suresi': f"{sum(self.ortalama_yanit_suresi)/max(1,len(self.ortalama_yanit_suresi)):.2f}s",
            'uptime': str(uptime).split('.')[0] if uptime else "0:00:00",
            'kategori_dagilimi': self.kategori_dagilimi,
            'database_boyutu': self.retriever.koleksiyon.count() if self.retriever.koleksiyon else 0,
            'chunking_kullanim_sayisi': self.chunking_kullanim_sayisi,
            'chunking_oran': chunking_oran,
            'chunk_istatistikleri': self.chunk_istatistikleri
        }

# 🎨 MODERN UI + OKUNAKLIK DÜZELTMELERİ
st.set_page_config(
    page_title="CPR Chunking System 🧩",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔧 UI OKUNAKLIK DÜZELTMELERİ - BEYAZLARDAKİ SIYAH METİN
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
    
    /* 🔧 BEYAZ KARTLARDA SİYAH METİN - OKUNAKLIK DÜZELTMESİ */
    .metric-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
        color: #212529 !important; /* SİYAH METİN */
    }
    
    .metric-card h4, .metric-card p, .metric-card li, .metric-card strong {
        color: #212529 !important; /* TÜM METİNLER SİYAH */
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* 🔧 BAŞARI KARTINDA KOYU YEŞİL METİN */
    .success-card {
        background: linear-gradient(145deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
        color: #155724 !important; /* KOYU YEŞİL */
    }
    
    .success-card strong {
        color: #0f4229 !important; /* DAHA KOYU YEŞİL */
    }
    
    /* 🔧 UYARI KARTINDA KOYU SARI METİN */
    .warning-card {
        background: linear-gradient(145deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffeaa7;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
        color: #856404 !important; /* KOYU SARI */
    }
    
    .warning-card strong {
        color: #664d03 !important; /* DAHA KOYU SARI */
    }
    
    /* 🔧 ACİL KART KOYU KIRMIZI METİN */
    .emergency-card {
        background: linear-gradient(145deg, #f8d7da, #f1aeb5);
        border: 1px solid #f1aeb5;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2);
        animation: pulse 2s infinite;
        color: #721c24 !important; /* KOYU KIRMIZI */
    }
    
    .emergency-card strong {
        color: #491217 !important; /* DAHA KOYU KIRMIZI */
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2); }
        50% { box-shadow: 0 6px 25px rgba(220, 53, 69, 0.4); }
        100% { box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2); }
    }
    
    /* 🔧 CHUNKING ÖZEL KARTI */
    .chunking-card {
        background: linear-gradient(145deg, #e3f2fd, #bbdefb);
        border: 1px solid #90caf9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.2);
        color: #0d47a1 !important; /* KOYU MAVİ */
    }
    
    .chunking-card strong, .chunking-card h4 {
        color: #0d47a1 !important;
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
    
    /* 🔧 DATAFRAME OKUNAKLIK DÜZELTMESİ */
    .stDataFrame {
        background-color: white !important;
    }
    
    .stDataFrame table {
        color: #212529 !important; /* SİYAH METİN */
    }
    
    .stDataFrame th {
        background-color: #f8f9fa !important;
        color: #495057 !important; /* KOYU GRİ BAŞLIK */
        font-weight: bold !important;
    }
    
    .stDataFrame td {
        color: #212529 !important; /* SİYAH HÜCRE METNİ */
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
        color: #212529 !important; /* SİYAH INPUT METNİ */
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
    <h1>🧩 CPR Query Chunking System v2.1</h1>
    <p><strong>Uzun Sorgu Desteği • Akıllı Parçalama • Gelişmiş UI</strong></p>
</div>
""", unsafe_allow_html=True)

# Gelişmiş örnek sorular
chunking_ornekleri = [
    "Yetişkinlerde kardiyak arrest durumunda epinefrin dozu ve uygulama şekli nedir?",
    "CPR sırasında kompresyon oranı, derinliği ve hızı nasıl olmalıdır?",
    "AED kullanımında dikkat edilmesi gereken adımlar ve güvenlik önlemleri nelerdir?",
    "Çocuklarda kalp masajı tekniği, kompresyon derinliği ve ventilasyon oranı nasıldır?",
    "Hipotermik arrest durumunda uygulanan özel protokoller ve dikkat edilecek hususlar?",
    "Hamile hastalarda CPR uygulaması ve özel pozisyonlama teknikleri nelerdir?",
    "CPR sırasında yapılan yapay solunum ile ilgili enfeksiyon riskleri ve korunma yöntemleri",
    "İleri yaşam desteğinde kullanılan ilaçların dozları ve uygulama zamanları"
]

# SIDEBAR - CHUNKING ÖZELLİKLERİYLE
with st.sidebar:
    st.markdown("## 🧩 CPR Chunking Hub")
    st.markdown("---")
    
    # Chunking bilgisi
    st.markdown("""
    <div class="chunking-card">
        <h4>🧩 Query Chunking Nedir?</h4>
        <p><strong>Uzun sorularınızı</strong> otomatik olarak parçalara böler ve her parça için ayrı arama yapar. Sonuçları birleştirerek <strong>daha doğru yanıtlar</strong> verir.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hızlı erişim butonları
    st.markdown("### 🚀 Chunking Test Soruları")
    for i, soru in enumerate(chunking_ornekleri):
        kisa_soru = soru[:35] + "..." if len(soru) > 35 else soru
        if st.button(kisa_soru, key=f"chunk_btn_{i}", width='stretch'):
            st.session_state.chunk_soru_input = soru
    
    st.markdown("---")
    
    # Sistem durumu
    st.markdown("### ⚙️ Sistem Durumu")
    
    if CHROMA_AVAILABLE and TRANSFORMERS_AVAILABLE:
        st.success("✅ Chunking System Aktif")
        st.info("🧠 Model: distiluse-multilingual")
        st.info("🧩 Chunking: Akıllı Parçalama")
        st.info("🎯 Eşik: Dinamik")
    else:
        st.error("❌ Kütüphaneler Eksik")
        st.code("pip install chromadb sentence-transformers plotly")
    
    st.markdown("---")
    
    # Chunking özellikler
    st.markdown("### 🏆 Chunking Özellikleri")
    st.markdown("""
    • 🧩 **Otomatik Chunking** (8+ kelime)
    • 🎯 **Multi-Chunk Bonus**
    • 📊 **Dinamik Eşikler**
    • 🔍 **Mega Kelime Genişletme**
    • ⚡ **Parallel Processing**
    • 📈 **Chunk İstatistikleri**
    • 🎨 **Enhanced UI/UX**
    • 🔧 **Okunabilirlik Fix**
    """)

# Sistem başlatma
if "chunk_sistem" not in st.session_state:
    with st.spinner("🧩 Query Chunking sistemi başlatılıyor..."):
        st.session_state.chunk_sistem = ChunkingCPRSistemi()
        st.session_state.chunk_basladi = st.session_state.chunk_sistem.sistem_baslat()

# Ana içerik
if not st.session_state.chunk_basladi:
    st.markdown("""
    <div class="warning-card">
        <h3>❌ Chunking Sistemi Başlatılamadı</h3>
        <p><strong>Kontrol listesi:</strong></p>
        <ul>
            <li><strong>cpr_egitim_bilgi_bankasi.json</strong> dosyası var mı?</li>
            <li>Kütüphaneler kurulu mu? <code>pip install chromadb sentence-transformers plotly</code></li>
            <li>İnternet bağlantısı aktif mi? (Model indirme)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="success-card">🧩 <strong>Query Chunking sistemi aktif!</strong> Uzun sorularınız otomatik parçalanacak.</div>', unsafe_allow_html=True)
    
    # CHUNKING METRİKLERİ DASHBOARD
    metrikliler = st.session_state.chunk_sistem.get_sistem_metrikleri()
    
    st.markdown("## 📊 Chunking Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🧩 Toplam Sorgu",
            value=metrikliler['toplam_sorgu'],
            delta=f"Chunking: {metrikliler['chunking_kullanim_sayisi']}"
        )
        
    with col2:
        st.metric(
            label="📈 Başarı Oranı",
            value=metrikliler['basari_orani'],
            delta=f"Chunking Oranı: {metrikliler['chunking_oran']}"
        )
    
    with col3:
        st.metric(
            label="⚡ Ortalama Yanıt",
            value=metrikliler['ortalama_yanit_suresi'],
            delta="Enhanced Speed"
        )
    
    with col4:
        chunk_stats = metrikliler['chunk_istatistikleri']
        ortalama_chunk = f"{chunk_stats['ortalama_chunk_per_sorgu']:.1f}"
        st.metric(
            label="🧩 Avg Chunk/Query",
            value=ortalama_chunk,
            delta=f"Toplam: {chunk_stats['toplam_chunk']}"
        )
    
    # Chunking performans detayları
    if metrikliler['chunking_kullanim_sayisi'] > 0:
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chunking vs Normal karşılaştırma grafiği
            chunking_data = {
                'Tip': ['Normal Sorgu', 'Chunking Sorgu'],
                'Sayı': [metrikliler['toplam_sorgu'] - metrikliler['chunking_kullanim_sayisi'], 
                        metrikliler['chunking_kullanim_sayisi']],
                'Renk': ['#ffa500', '#667eea']
            }
            
            fig = px.bar(
                x=chunking_data['Tip'],
                y=chunking_data['Sayı'],
                color=chunking_data['Tip'],
                title="📊 Chunking vs Normal Sorgu Dağılımı",
                color_discrete_map={
                    'Normal Sorgu': '#ffa500',
                    'Chunking Sorgu': '#667eea'
                }
            )
            fig.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🧩 Chunking İstatistikleri")
            chunk_stats = metrikliler['chunk_istatistikleri']
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>📊 Toplam Chunk:</strong> {chunk_stats['toplam_chunk']}<br>
                <strong>✅ Başarılı Chunk:</strong> {chunk_stats['basarili_chunk']}<br>
                <strong>📈 Ortalama/Sorgu:</strong> {chunk_stats['ortalama_chunk_per_sorgu']:.1f}<br>
                <strong>🎯 Chunking Oranı:</strong> {metrikliler['chunking_oran']}
            </div>
            """, unsafe_allow_html=True)
    
    # Kategori dağılımı (düzeltilmiş renk)
    if metrikliler['kategori_dagilimi']:
        st.markdown("---")
        st.markdown("### 📂 CPR Kategori Dağılımı")
        
        fig = px.pie(
            values=list(metrikliler['kategori_dagilimi'].values()),
            names=[k.replace('_', ' ').title() for k in metrikliler['kategori_dagilimi'].keys()],
            title="CPR Doküman Kategorileri",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textfont_color="black")  # 🔧 SİYAH METİN
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performans trend grafiği
    performans_data = st.session_state.chunk_sistem.retriever.get_performans_grafigi()
    if performans_data:
        st.markdown("---")
        st.markdown("### 📈 Chunking Performans Trendi")
        
        fig = go.Figure()
        
        # Skor çizgisi
        fig.add_trace(go.Scatter(
            x=list(range(len(performans_data['skorlar']))),
            y=performans_data['skorlar'],
            mode='lines+markers',
            name='Benzerlik Skoru',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#764ba2'),
            hovertemplate='<b>%{text}</b><br>Skor: %{y:.3f}<br>Chunk: %{customdata}<extra></extra>',
            text=performans_data['sorgular'],
            customdata=performans_data['chunk_sayilari']
        ))
        
        # Chunk sayısı bar
        fig.add_trace(go.Bar(
            x=list(range(len(performans_data['chunk_sayilari']))),
            y=performans_data['chunk_sayilari'],
            name='Chunk Sayısı',
            yaxis='y2',
            marker_color='rgba(255, 165, 0, 0.6)',
            hovertemplate='<b>%{text}</b><br>Chunk: %{y}<extra></extra>',
            text=performans_data['sorgular']
        ))
        
        fig.update_layout(
            title="Son Sorguların Performans ve Chunk Analizi",
            xaxis_title="Sorgu Sırası",
            yaxis_title="Benzerlik Skoru",
            yaxis2=dict(
                title="Chunk Sayısı",
                overlaying='y',
                side='right',
                range=[0, max(performans_data['chunk_sayilari']) + 1]
            ),
            template="plotly_white",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # CHUNKING SORGU BÖLÜMÜ
    st.markdown("## 💬 Akıllı Chunking Sorgulama")
    
    # Chunking açıklama kutusu
    st.markdown("""
    <div class="chunking-card">
        <h4>🧩 Nasıl Çalışır?</h4>
        <p><strong>8+ kelimelik sorular</strong> otomatik olarak parçalara bölünür. Örneğin:</p>
        <p><em>"AED kullanımında dikkat edilmesi gereken adımlar ve güvenlik önlemleri"</em></p>
        <p>↓</p>
        <p><strong>Chunk 1:</strong> "AED kullanımında dikkat edilmesi gereken"<br>
        <strong>Chunk 2:</strong> "adımlar ve güvenlik önlemleri"<br>
        <strong>Chunk 3:</strong> "CPR AED kullanımı" (otomatik eklenen)</p>
    </div>
    """, unsafe_allow_html=True)
    
    soru = st.text_input(
        "CPR konusunda uzun ve detaylı sorunuzu yazın:",
        value=st.session_state.get('chunk_soru_input', ''),
        placeholder="Örn: Yetişkinlerde kardiyak arrest durumunda epinefrin dozı, uygulama şekli ve dikkat edilmesi gereken yan etkiler nelerdir?",
        key="chunk_ana_input",
        help="💡 İpucu: 8+ kelimelik sorular otomatik chunking alır"
    )
    
    # Chunking preview
    if soru.strip():
        kelime_sayisi = len(soru.split())
        if kelime_sayisi > 8:
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #e8f5e8, #d4f1d4); padding: 1rem; border-radius: 8px; margin: 1rem 0; color: #2e7d32;">
                <strong>🧩 Chunking Aktif:</strong> {kelime_sayisi} kelimelik sorgu parçalara bölünecek
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #fff3e0, #ffecb3); padding: 1rem; border-radius: 8px; margin: 1rem 0; color: #f57f17;">
                <strong>📝 Normal Arama:</strong> {kelime_sayisi} kelimelik sorgu için chunking kullanılmayacak
            </div>
            """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        sorgula_btn = st.button("🧩 Chunking Analizi", type="primary", width='stretch')
    
    with col2:
        if st.button("🔄 Temizle", width='stretch'):
            st.session_state.chunk_soru_input = ""
            st.rerun()
    
    with col3:
        if st.button("🎲 Rastgele Uzun", width='stretch'):
            st.session_state.chunk_soru_input = random.choice(chunking_ornekleri)
            st.rerun()
    
    # CHUNKING SONUÇLARI
    if sorgula_btn and soru.strip():
        with st.spinner("🧩 Akıllı chunking sistemi analiz yapıyor..."):
            time.sleep(0.8)  # Professional feel
            sonuc = st.session_state.chunk_sistem.chunking_sorgulama(soru)
        
        st.markdown("---")
        
        # Chunking durumu kartı
        if sonuc["chunking_kullanildi"]:
            st.markdown('<div class="chunking-card">🧩 <strong>Query Chunking Kullanıldı!</strong> Uzun sorgunuz parçalara bölündü ve analiz edildi.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background: linear-gradient(145deg, #f3e5f5, #e1bee7); padding: 1rem; border-radius: 12px; margin: 1rem 0; color: #4a148c;">📝 <strong>Normal Arama:</strong> Kısa sorgu için chunking kullanılmadı.</div>', unsafe_allow_html=True)
        
        # Sonuç durumu kartı
        if sonuc["basarili"]:
            if sonuc["acil_durum"]:
                st.markdown('<div class="emergency-card">🚨 <strong>KRİTİK ACİL PROTOKOL BULUNDU!</strong> Yaşamsal öneme sahip bilgi.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-card">✅ <strong>Chunking başarılı!</strong> Yüksek kaliteli protokol sonuçları bulundu.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-card">⚠️ <strong>Chunking tamamlandı ancak yeterli eşleşme bulunamadı.</strong> Akıllı öneriler sunuluyor.</div>', unsafe_allow_html=True)
        
        # Ana yanıt
        st.markdown(sonuc['yanit'])
        
        # CHUNKING ANALİZ RAPORU
        st.markdown("---")
        st.markdown("## 📊 Chunking Analiz Raporu")
        
        # Üç sütunlu analiz
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔍 Arama Detayları")
            st.markdown(f"""
            <div class="metric-card">
                <strong>🎯 En İyi Skor:</strong> {sonuc['en_iyi_skor']:.3f}<br>
                <strong>🔍 Bulunan Doküman:</strong> {sonuc['bulunan_dokuman_sayisi']}<br>
                <strong>⭐ Kaliteli Sonuç:</strong> {sonuc['kaliteli_sonuc_sayisi']}<br>
                <strong>🎚️ Kullanılan Eşik:</strong> {sonuc['kullanilan_esik']:.3f}<br>
                <strong>📝 Eşik Tipi:</strong> {sonuc['esik_tipi'].title()}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🧩 Chunking İstatistikleri")
            chunk_stats = sonuc['chunking_istatistikleri']
            st.markdown(f"""
            <div class="metric-card">
                <strong>🧩 Chunking Kullanıldı:</strong> {'✅ Evet' if sonuc['chunking_kullanildi'] else '❌ Hayır'}<br>
                <strong>⚡ Yanıt Süresi:</strong> {sonuc['yanit_suresi']:.2f}s<br>
                <strong>📊 Toplam Chunk:</strong> {chunk_stats['toplam_chunk']}<br>
                <strong>📈 Avg Chunk/Q:</strong> {chunk_stats['ortalama_chunk_per_sorgu']:.1f}<br>
                <strong>✅ Başarılı Chunk:</strong> {chunk_stats['basarili_chunk']}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("### 📈 Sistem Performansı")
            st.markdown(f"""
            <div class="metric-card">
                <strong>🏆 Performans:</strong> {sonuc['sistem_performansi']}<br>
                <strong>📊 Genel Başarı:</strong> {sonuc['basari_orani']}<br>
                <strong>🫀 CPR Odaklı:</strong> {'✅ Evet' if sonuc['cpr_odakli'] else '❌ Hayır'}<br>
                <strong>🚨 Acil Durum:</strong> {'🔴 EVET' if sonuc['acil_durum'] else '🟢 Normal'}<br>
                <strong>🎯 Sorgu Karmaşıklığı:</strong> {'Yüksek' if sonuc['chunking_kullanildi'] else 'Basit'}
            </div>
            """, unsafe_allow_html=True)
        
        # Detaylı eşleşme tablosu - 🔧 OKUNAKLIK DÜZELTİLMİŞ
        if sonuc["sonuc_detaylari"]:
            st.markdown("---")
            st.markdown("### 🎯 En İyi Eşleşmeler - Chunking Detay Analizi")
            
            tablo_data = []
            for i, (final_skor, ham_max_skor, chunk_sayisi, kategori, guvenilirlik) in enumerate(sonuc["sonuc_detaylari"]):
                # Chunking bonus hesapla
                if chunk_sayisi > 1:
                    multi_bonus = f"+{((chunk_sayisi-1)*15):.0f}%"
                else:
                    multi_bonus = "Yok"
                
                # Kalite değerlendirmesi
                if final_skor > 0.7:
                    kalite = "🏆 Mükemmel"
                    renk = "🟢"
                elif final_skor > 0.5:
                    kalite = "🚀 Çok İyi"
                    renk = "🔵"
                elif final_skor > 0.3:
                    kalite = "📈 İyi"
                    renk = "🟡"
                else:
                    kalite = "📊 Orta"
                    renk = "🟠"
                
                tablo_data.append({
                    "🔍": f"#{i+1}",
                    "Durum": renk,
                    "🏆 Final Skor": f"{final_skor:.3f}",
                    "📊 Ham Skor": f"{ham_max_skor:.3f}",
                    "🧩 Chunk Sayısı": chunk_sayisi,
                    "🎁 Multi Bonus": multi_bonus,
                    "📂 Kategori": kategori.replace('_', ' ').title(),
                    "⭐ Güvenilirlik": f"%{guvenilirlik*100:.0f}",
                    "🎯 Kalite": kalite
                })
            
            # Okunabilirlik düzeltilmiş tablo
            st.dataframe(
                tablo_data, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "🔍": st.column_config.TextColumn(width="small"),
                    "Durum": st.column_config.TextColumn(width="small"),
                    "🏆 Final Skor": st.column_config.NumberColumn(format="%.3f"),
                    "📊 Ham Skor": st.column_config.NumberColumn(format="%.3f"),
                    "🧩 Chunk Sayısı": st.column_config.NumberColumn(format="%d")
                }
            )
    
    elif sorgula_btn and not soru.strip():
        st.markdown('<div class="warning-card">❗ <strong>Lütfen bir CPR sorusu yazın.</strong> Uzun sorular için chunking özelliği kullanılacak.</div>', unsafe_allow_html=True)

# FOOTER - CHUNKING ÖZELLİKLİ
st.markdown("---")
st.markdown("## 🎓 CPR Chunking Teknoloji Merkezi")

# Özellik showcase
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>🧩 Query Chunking Teknolojisi</h4>
        <ul>
            <li><strong>Otomatik sorgu parçalama</strong></li>
            <li><strong>Multi-chunk bonus sistemi</strong></li>
            <li><strong>Parallel processing</strong></li>
            <li><strong>Intelligent chunk weighting</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>🎯 Gelişmiş Arama Sistemi</h4>
        <ul>
            <li><strong>Dinamik eşik değerleri</strong></li>
            <li><strong>Mega kelime genişletme</strong></li>
            <li><strong>Context-aware scoring</strong></li>
            <li><strong>Real-time chunk analysis</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>📊 Performans & UI İyileştirmeleri</h4>
        <ul>
            <li><strong>Okunabilirlik düzeltmeleri</strong></li>
            <li><strong>Chunk performans grafikleri</strong></li>
            <li><strong>Enhanced color contrast</strong></li>
            <li><strong>Professional dashboard</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Final professional card
st.markdown("""
<div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 20px; color: white; margin: 2rem 0; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);">
    <h2>🧩 CPR Query Chunking System v2.1</h2>
    <p style="font-size: 1.2em; margin: 1rem 0;"><strong>Uzun sorularınızı parçalara bölen akıllı teknoloji</strong></p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap;">
        <div>🧩 <strong>Auto Chunking</strong></div>
        <div>🎯 <strong>Multi-Chunk Bonus</strong></div>
        <div>📊 <strong>Dynamic Thresholds</strong></div>
        <div>⚡ <strong>Enhanced Speed</strong></div>
        <div>🔧 <strong>UI Fixed</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px; margin: 1rem 0; color: #212529;">
    <p><strong>⚠️ UYARI:</strong> Bu sistem eğitim amaçlıdır. Gerçek acil durumlarda <strong>112</strong>'yi arayın ve profesyonel tıbbi yardım alın.</p>
    <p>Query Chunking teknolojisi ile <strong>uzun ve karmaşık sorularınız</strong> daha doğru yanıtlanır.</p>
</div>
""", unsafe_allow_html=True)