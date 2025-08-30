# Düzeltilmiş Chroma Vector Database CPR Sistemi
# Bu dosyayı saglik_rock_chroma.py olarak kaydet

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import uuid

# Vector Database
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("[UYARI] ChromaDB bulunamadi. Kurulum: pip install chromadb")

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[UYARI] Sentence Transformers bulunamadi.")

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
            print("[HATA] Gerekli kütüphaneler eksik!")
            return False
        
        try:
            # ChromaDB client oluştur
            print("[BASLANGIC] ChromaDB baglanıyor...")
            self.chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            
            # Model yükle
            print("[BASLANGIC] Sentence Transformer yukleniyor...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Koleksiyon oluştur veya getir
            try:
                self.koleksiyon = self.chroma_client.get_collection(self.koleksiyon_adi)
                print(f"[BASARI] Mevcut koleksiyon yuklendi: {self.koleksiyon.count()} dokuman")
            except:
                self.koleksiyon = self.chroma_client.create_collection(
                    name=self.koleksiyon_adi,
                    metadata={"description": "CPR egitim vektör veritabanı"}
                )
                print("[BASARI] Yeni koleksiyon olusturuldu")
            
            return True
            
        except Exception as e:
            print(f"[HATA] Sistem baslatma hatasi: {str(e)}")
            return False
    
    def dokumanlar_ekle(self, dokumanlar: List[Dict], temizle: bool = False):
        """Dökümanları vector database'e ekle"""
        if not self.koleksiyon or not self.model:
            print("[HATA] Sistem baslatilmamis!")
            return False
        
        if temizle:
            # Mevcut koleksiyonu temizle
            self.chroma_client.delete_collection(self.koleksiyon_adi)
            self.koleksiyon = self.chroma_client.create_collection(
                name=self.koleksiyon_adi,
                metadata={"description": "CPR egitim vektör veritabanı"}
            )
            print("[BILGI] Koleksiyon temizlendi")
        
        print(f"[ISLEME] {len(dokumanlar)} dokuman vector database'e ekleniyor...")
        
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
                'guvenilirlik': dok.get('guvenilirlik', 0.8),
                'acillik_seviyesi': dok.get('acillik_seviyesi', 'normal'),
                'kaynak': dok.get('metadata', {}).get('kaynak', 'bilinmiyor')
            }
            metadatas.append(metadata)
            
            # Tam döküman metni
            documents.append(icerik)
            
            if (i + 1) % 5 == 0:
                print(f"[EMBEDDING] {i + 1}/{len(dokumanlar)} dokuman islendi")
        
        # Batch olarak ekle
        try:
            self.koleksiyon.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
            print(f"[BASARI] {len(dokumanlar)} dokuman vector database'e eklendi!")
            return True
            
        except Exception as e:
            print(f"[HATA] Dokuman ekleme hatasi: {str(e)}")
            return False
    
    def hizli_arama(self, sorgu: str, top_k: int = 5, filtreler: Dict = None) -> List[Dict]:
        """ChromaDB ile ultra hızlı semantik arama"""
        if not self.koleksiyon or not self.model:
            print("[HATA] Sistem baslatilmamis!")
            return []
        
        try:
            # Sorgu embedding'i oluştur
            sorgu_embedding = self.model.encode(sorgu).tolist()
            
            # ChromaDB'den ara
            sonuclar = self.koleksiyon.query(
                query_embeddings=[sorgu_embedding],
                n_results=top_k,
                where=filtreler
            )
            
            # Sonuçları formatla
            arama_sonuclari = []
            if sonuclar['documents'] and len(sonuclar['documents'][0]) > 0:
                for i in range(len(sonuclar['documents'][0])):
                    sonuc = {
                        'id': sonuclar['ids'][0][i],
                        'icerik': sonuclar['documents'][0][i],
                        'benzerlik_skoru': max(0, 1 - sonuclar['distances'][0][i]),  # Düzeltme
                        'metadata': sonuclar['metadatas'][0][i],
                        'kategori': sonuclar['metadatas'][0][i].get('kategori', 'bilinmiyor'),
                        'guvenilirlik': sonuclar['metadatas'][0][i].get('guvenilirlik', 0.8)
                    }
                    arama_sonuclari.append(sonuc)
            
            print(f"[ARAMA] '{sorgu}' icin {len(arama_sonuclari)} sonuc bulundu")
            
            # Debug: İlk 3 sonucun skorlarını yazdır
            for i, sonuc in enumerate(arama_sonuclari[:3]):
                print(f"[DEBUG] Sonuc {i+1}: Skor={sonuc['benzerlik_skoru']:.3f}, Kategori={sonuc['kategori']}")
            
            return arama_sonuclari
            
        except Exception as e:
            print(f"[HATA] Arama hatasi: {str(e)}")
            return []

class HizliSaglikROCK:
    """ChromaDB ile hızlandırılmış CPR eğitim sistemi"""
    
    def __init__(self):
        self.retriever = ChromaRetriever()
        self.sorgu_sayisi = 0
        self.basarili_sorgu = 0
        
        # CPR özel anahtar kelimeler
        self.acil_kelimeler = [
            'cpr', 'kalp krizi', 'kalp durması', 'resüsitasyon',
            'defibrilasyon', 'aed', 'kompresyon', 'canlandırma'
        ]
        
        self.cpr_kategoriler = {
            'temel_cpr': ['cpr', 'kompresyon', 'solunum', 'nabız'],
            'aed': ['defibrilasyon', 'aed', 'şok', 'elektrot'],
            'ilaç': ['epinephrin', 'amiodarone', 'atropin', 'doz'],
            'pediatrik': ['bebek', 'çocuk', 'pediatrik']
        }
    
    def sistem_baslat(self, veri_dosyasi: str = "cpr_egitim_bilgi_bankasi.json"):
        """CPR eğitim sistemi başlatma"""
        print("[BASLANGIC] CPR Egitim sistemi baslatiliyor...")
        
        # ChromaDB başlat
        if not self.retriever.sistem_baslat():
            print("[HATA] Vector database baslatilamadi!")
            return False
        
        # CPR veri yükle
        try:
            with open(veri_dosyasi, 'r', encoding='utf-8') as f:
                veriler = json.load(f)
            
            print(f"[YUKLEME] {len(veriler)} CPR protokol dokumani yuklendi")
            
            # Vector database'e ekle
            mevcut_dok_sayisi = self.retriever.koleksiyon.count()
            if mevcut_dok_sayisi == 0:
                self.retriever.dokumanlar_ekle(veriler, temizle=True)
            else:
                print(f"[BILGI] Mevcut {mevcut_dok_sayisi} CPR dokumani kullaniliyor")
            
            print("[BASARI] CPR Egitim sistemi hazir!")
            return True
            
        except FileNotFoundError:
            print(f"[HATA] CPR veri dosyasi bulunamadi: {veri_dosyasi}")
            return False
        except Exception as e:
            print(f"[HATA] Sistem baslatma hatasi: {str(e)}")
            return False
    
    def soru_sor(self, soru: str) -> Dict:
        """CPR sorusu cevaplama"""
        self.sorgu_sayisi += 1
        
        print(f"\n[CPR SORGU {self.sorgu_sayisi}] {soru}")
        
        # Arama yap
        sonuclar = self.retriever.hizli_arama(soru, top_k=5)
        
        # Acil durum tespiti
        acil_durum = any(kelime in soru.lower() for kelime in self.acil_kelimeler)
        
        # Kaliteli sonuçları filtrele (eşik düşürüldü)
        kaliteli_sonuclar = [s for s in sonuclar if s['benzerlik_skoru'] > 0.1]
        
        if kaliteli_sonuclar:
            self.basarili_sorgu += 1
            yanit = self._cpr_yanit_olustur(soru, kaliteli_sonuclar[:3], acil_durum)
            basarili = True
        else:
            yanit = self._cpr_basarisiz_yanit(soru)
            basarili = False
        
        return {
            'soru': soru,
            'yanit': yanit,
            'basarili': basarili,
            'acil_durum': acil_durum,
            'bulunan_dokuman_sayisi': len(sonuclar),
            'sistem_performansi': f"{self.basarili_sorgu}/{self.sorgu_sayisi}",
            'sonuc_detaylari': [
                (s['benzerlik_skoru'], s['kategori'], s['id']) 
                for s in kaliteli_sonuclar[:3]
            ]
        }
    
    def _cpr_yanit_olustur(self, soru: str, sonuclar: List[Dict], acil: bool) -> str:
        """CPR yanıt şablonu"""
        if acil:
            başlık = "🚨 KRITIK CPR PROTOKOLÜ"
            uyari = "Bu yaşamsal durum! Protokolü kesin takip edin.\n\n"
        else:
            başlık = "📋 CPR EĞİTİM PROTOKOLÜ"
            uyari = ""
        
        icerik_blokları = []
        for sonuc in sonuclar:
            kategori = sonuc['kategori'].upper()
            güvenilirlik = sonuc['guvenilirlik']
            benzerlik = sonuc['benzerlik_skoru']
            
            icerik_blokları.append(
                f"[{kategori}] (Güvenilirlik: {güvenilirlik:.1f}, Eşleşme: {benzerlik:.2f})\n"
                f"{sonuc['icerik'][:400]}{'...' if len(sonuc['icerik']) > 400 else ''}"
            )
        
        return f"""{başlık}

{uyari}SORU: {soru}

PROTOKOL BİLGİLERİ:
{chr(10).join(icerik_blokları)}

*** EĞİTİM NOTU ***
AHA 2020 Guidelines temelinde hazırlanmıştır.
Gerçek uygulamada takım koordinasyonu kritik."""
    
    def _cpr_basarisiz_yanit(self, soru: str) -> str:
        return f"""📋 CPR EĞİTİM SİSTEMİ

SORU: {soru}

SONUÇ: Spesifik protokol bulunamadı.

ÖNERİLER:
• CPR, AED, epinephrin anahtar kelimelerini kullanın
• Yaş grubu belirtin (yetişkin/çocuk/bebek)
• Daha spesifik sorular sorun"""

# Test sistemi
if __name__ == "__main__":
    print("=== CPR EGİTİM SİSTEMİ TEST ===")
    
    # Sistem başlat
    cpr_sistem = HizliSaglikROCK()
    
    if cpr_sistem.sistem_baslat():
        print("\n=== CPR TEST SORULARI ===")
        
        test_sorulari = [
            "CPR oranı nedir?",
            "AED nasıl kullanılır?",
            "Çocuk CPR protokolü?",
            "Epinephrin dozu nedir?"
        ]
        
        for soru in test_sorulari:
            sonuc = cpr_sistem.soru_sor(soru)
            print(f"\n{'='*50}")
            print(f"SORU: {sonuc['soru']}")
            print(f"BAŞARILI: {'EVET' if sonuc['basarili'] else 'HAYIR'}")
            print(f"BULUNAN: {sonuc['bulunan_dokuman_sayisi']} döküman")
            print(f"YANIT: {sonuc['yanit'][:200]}...")
        
        print(f"\n=== PERFORMANS ===")
        print(f"Başarı: {cpr_sistem.basarili_sorgu}/{cpr_sistem.sorgu_sayisi}")
    
    else:
        print("[HATA] Sistem başlatılamadı!")
        print("1. pip install chromadb sentence-transformers")
        print("2. cpr_egitim_bilgi_bankasi.json dosyasının varlığını kontrol edin")