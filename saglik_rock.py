# Gelişmiş Sağlık ROCK Sistemi - Sentence Transformers ile
# Çok daha akıllı ve doğru arama yapabilen versiyon

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

# Sentence Transformers kütüphanesi - akıllı arama için
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
    print("[BASARI] Sentence Transformers yuklendi - akilli arama aktif!")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[UYARI] Sentence Transformers bulunamadi. Basit arama kullanilacak.")
    print("Kurulum icin: pip install sentence-transformers")

@dataclass
class GelismisTimbiDokuman:
    """Gelişmiş tıbbi dökümanlar için veri yapısı"""
    id: str
    icerik: str
    tibbi_kategori: str
    guvenilirlik_skoru: float
    # Sentence transformer embeddingi (384 boyut)
    semantic_embedding: Optional[np.ndarray] = None
    # Eski basit embedding (yedek olarak)
    simple_embedding: Optional[np.ndarray] = None
    metadata: Dict = None

class AkilliTibbiRetriever:
    """Sentence Transformers ile akıllı tıbbi arama"""
    
    def __init__(self):
        self.dokumanlar: List[GelismisTimbiDokuman] = []
        
        # Sentence Transformer modeli - Türkçe destekli
        if TRANSFORMERS_AVAILABLE:
            print("[YUKLENIYOR] Sentence Transformer modeli...")
            # Türkçe için en iyi model: multilingual-MiniLM
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("[BASARI] Akilli arama modeli yuklendi!")
        else:
            self.model = None
            # Basit arama için yedek sistem
            self.tibbi_terimler: Dict[str, int] = {}
    
    def dokumanlar_ekle(self, docs: List[Dict[str, str]]):
        """Dökümanları sisteme ekle ve embeddingleri oluştur"""
        print(f"[ISLENIYOR] {len(docs)} dokuman ekleniyor...")
        
        for i, doc in enumerate(docs):
            yeni_dok = GelismisTimbiDokuman(
                id=doc.get('id', str(i)),
                icerik=doc['icerik'],
                tibbi_kategori=doc.get('kategori', 'genel'),
                guvenilirlik_skoru=doc.get('guvenilirlik', 0.8),
                metadata=doc.get('metadata', {})
            )
            
            # Sentence Transformer embedding oluştur
            if self.model:
                print(f"[EMBEDDING] Dokuman {i+1} isleniyor...")
                yeni_dok.semantic_embedding = self.model.encode(yeni_dok.icerik)
            else:
                # Basit embedding yedek sistem
                yeni_dok.simple_embedding = self._basit_embedding_olustur(yeni_dok.icerik)
            
            self.dokumanlar.append(yeni_dok)
        
        print(f"[TAMAMLANDI] {len(docs)} dokuman basariyla eklendi!")
    
    def _basit_embedding_olustur(self, metin: str) -> np.ndarray:
        """Yedek basit embedding sistemi"""
        kelimeler = metin.lower().split()
        
        # Kelime dağarcığı oluştur
        if not hasattr(self, 'tibbi_terimler') or not self.tibbi_terimler:
            tum_kelimeler = set()
            for dok in self.dokumanlar:
                tum_kelimeler.update(dok.icerik.lower().split())
            if not tum_kelimeler:  # İlk döküman ekleniyor
                tum_kelimeler = set(kelimeler)
            self.tibbi_terimler = {kelime: i for i, kelime in enumerate(tum_kelimeler)}
        
        # Embedding vektörü
        vektor = np.zeros(len(self.tibbi_terimler))
        for kelime in kelimeler:
            if kelime in self.tibbi_terimler:
                vektor[self.tibbi_terimler[kelime]] += 1
        
        norm = np.linalg.norm(vektor)
        return vektor / norm if norm > 0 else vektor
    
    def akilli_arama(self, sorgu: str, top_k: int = 5) -> List[Tuple[float, GelismisTimbiDokuman]]:
        """Sentence Transformers ile semantik arama"""
        if not self.dokumanlar:
            return []
        
        if self.model:
            # Akıllı semantik arama
            print(f"[ARAMA] '{sorgu}' için semantik arama yapiliyor...")
            sorgu_embedding = self.model.encode(sorgu)
            
            skorlar = []
            for dokuman in self.dokumanlar:
                # Cosine similarity hesapla
                benzerlik = np.dot(sorgu_embedding, dokuman.semantic_embedding)
                # Güvenilirlik skorunu da ekle
                final_skor = benzerlik * dokuman.guvenilirlik_skoru
                skorlar.append((final_skor, dokuman))
            
        else:
            # Basit arama yedek sistem
            print(f"[ARAMA] '{sorgu}' için basit arama yapiliyor...")
            sorgu_embedding = self._basit_embedding_olustur(sorgu)
            
            skorlar = []
            for dokuman in self.dokumanlar:
                if dokuman.simple_embedding is None:
                    dokuman.simple_embedding = self._basit_embedding_olustur(dokuman.icerik)
                
                benzerlik = np.dot(sorgu_embedding, dokuman.simple_embedding)
                final_skor = benzerlik * dokuman.guvenilirlik_skoru
                skorlar.append((final_skor, dokuman))
        
        # En yüksek skorluları döndür
        skorlar.sort(reverse=True, key=lambda x: x[0])
        print(f"[SONUC] En iyi {min(top_k, len(skorlar))} sonuc bulundu")
        
        # Skorları da döndür - debug için yararlı
        return skorlar[:top_k]

class GelismisKontrolMekanizmasi:
    """Gelişmiş güvenlik kontrolü"""
    
    def __init__(self, minimum_guvenilirlik: float = 0.7, minimum_benzerlik: float = 0.3):
        self.minimum_guvenilirlik = minimum_guvenilirlik
        self.minimum_benzerlik = minimum_benzerlik  # Çok düşük benzerlik skorlarını filtrele
        
        self.hassas_konular = [
            'acil', 'kalp krizi', 'felç', 'zehirlenme', 'koma',
            'kanser', 'ameliyat', 'doktor', 'ölüm', 'kan', 'göğüs ağrısı'
        ]
    
    def gelismis_filtreleme(self, skorlu_dokumanlar: List[Tuple[float, GelismisTimbiDokuman]], 
                           sorgu: str) -> str:
        """Gelişmiş içerik filtreleme ve skora dayalı seçim"""
        
        if not skorlu_dokumanlar:
            return ""
        
        print(f"[FILTRELEME] {len(skorlu_dokumanlar)} dokuman filtreleniyor...")
        
        # Acil durum tespiti
        acil_durum = any(hassas in sorgu.lower() for hassas in self.hassas_konular)
        
        # Skorları yazdır - debug için
        for i, (skor, dok) in enumerate(skorlu_dokumanlar[:3]):
            print(f"[SKOR {i+1}] {skor:.3f} - {dok.tibbi_kategori} - {dok.icerik[:50]}...")
        
        guvenli_icerik = []
        for skor, dokuman in skorlu_dokumanlar:
            # Güvenilirlik ve benzerlik kontrolü
            if (dokuman.guvenilirlik_skoru >= self.minimum_guvenilirlik and 
                skor >= self.minimum_benzerlik):
                
                kisaltilmis_icerik = (dokuman.icerik[:400] + "..." 
                                    if len(dokuman.icerik) > 400 
                                    else dokuman.icerik)
                
                guvenli_icerik.append(
                    f"[Kategori: {dokuman.tibbi_kategori}] [Benzerlik: {skor:.2f}]\n"
                    f"{kisaltilmis_icerik}\n"
                    f"[Guvenilirlik: {dokuman.guvenilirlik_skoru}/1.0]"
                )
        
        # Acil durum uyarısı
        if acil_durum:
            uyari = "[!!! ACIL DURUM UYARISI !!!]\nBu ciddi bir durum olabilir. Derhal 112'yi arayin!\n\n"
            return uyari + "\n\n".join(guvenli_icerik[:2])
        
        return "\n\n".join(guvenli_icerik[:3])

class GelismissSaglikROCK:
    """Sentence Transformers ile güçlendirilmiş sağlık sistemi"""
    
    def __init__(self):
        self.retriever = AkilliTibbiRetriever()
        self.kontrol = GelismisKontrolMekanizmasi()
        self.bilgi_bankasi: List[Dict] = []
        
        # Performans istatistikleri
        self.sorgu_sayisi = 0
        self.basarili_sorgu = 0
    
    def tibbi_bilgi_yukle(self, dosya_yolu: str):
        """Tıbbi bilgi bankasını yükle"""
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            self.bilgi_bankasi = json.load(f)
        
        print(f"[YUKLEME] {len(self.bilgi_bankasi)} tibbi dokuman yukleniyor...")
        self.retriever.dokumanlar_ekle(self.bilgi_bankasi)
    
    def tibbi_soru_cevapla(self, soru: str, detayli_log: bool = False) -> Dict[str, str]:
        """Gelişmiş ROCK pipeline ile soru cevaplama"""
        self.sorgu_sayisi += 1
        
        if detayli_log:
            print(f"\n[SORGU {self.sorgu_sayisi}] {soru}")
        
        # 1. Akıllı retrieval
        skorlu_sonuclar = self.retriever.akilli_arama(soru, top_k=5)
        
        # 2. Gelişmiş kontrol
        filtrelenmis_icerik = self.kontrol.gelismis_filtreleme(skorlu_sonuclar, soru)
        
        # 3. Yanıt üretimi
        if filtrelenmis_icerik:
            yanit = self._profesyonel_yanit_uret(soru, filtrelenmis_icerik)
            self.basarili_sorgu += 1
        else:
            yanit = self._bulunamadi_yaniti()
        
        # İstatistikler
        en_yuksek_skor = skorlu_sonuclar[0][0] if skorlu_sonuclar else 0.0
        
        return {
            "soru": soru,
            "bulunan_dokuman_sayisi": len(skorlu_sonuclar),
            "en_yuksek_benzerlik_skoru": f"{en_yuksek_skor:.3f}",
            "icerik_ozeti": filtrelenmis_icerik[:200] + "..." if len(filtrelenmis_icerik) > 200 else filtrelenmis_icerik,
            "tibbi_yanit": yanit,
            "sistem_performansi": f"{self.basarili_sorgu}/{self.sorgu_sayisi}"
        }
    
    def _profesyonel_yanit_uret(self, soru: str, icerik: str) -> str:
        """Profesyonel tıbbi yanıt şablonu"""
        return f"""=== TIBBI BILGI SISTEMI ===

SORU: {soru}

BILGI BANKASI ANALIZI:
{icerik}

*** ONEMLI TIBBI UYARI ***
- Bu bilgiler sadece genel bilgilendirme amaclidir
- Kesin tani ve tedavi icin mutlaka hekim gorunuz  
- Acil durumlarda 112'yi arayin
- Ilac kullanimi oncesi doktor onayini alin

=== Saglikli gunler dileriz ==="""
    
    def _bulunamadi_yaniti(self) -> str:
        """Bilgi bulunamadığında standart yanıt"""
        return """=== TIBBI BILGI SISTEMI ===

SONUC: İlgili konuda guvenilir tibbi bilgi bulunamadi.

ONERILER:
- Sorunuzu daha detayli ifade edin
- Alternatif kelimeler kullanin  
- Mutlaka bir saglik uzmani ile gorusun

ACIL DURUMLARDA: 112'yi arayin

=== Saglikli gunler dileriz ==="""

def gelismis_saglik_bilgi_bankasi_olustur():
    """Daha kapsamlı test bilgi bankası"""
    kapsamli_tibbi_dokumanlar = [
        {
            "id": "1",
            "icerik": "Ateş vücut sıcaklığının normalin üzerine çıkmasıdır. 37.5°C üzeri ateş sayılır. Yüksek ateş, enfeksiyon belirtisi olabilir. Bol sıvı alın, dinlenin. 38.5°C üzeri ateşte doktora başvurun. Parasetamol veya ibuprofen ateş düşürücü olarak kullanılabilir.",
            "kategori": "semptom",
            "guvenilirlik": 0.9,
            "metadata": {"kaynak": "Sağlık Bakanlığı", "guncelleme": "2024"}
        },
        {
            "id": "2",
            "icerik": "Baş ağrısı çok yaygın bir şikayettir. Migren, gerilim tipi baş ağrısı, sinüzit kaynaklı olabilir. Stres, uykusuzluk, dehidratasyon, göz yorgunluğu nedeniyle başınız ağrıyabilir. Kafa ağrısı için parasetamol, ibuprofen etkilidir. Sürekli şiddetli baş ağrılarında doktor kontrolü şarttır.",
            "kategori": "semptom", 
            "guvenilirlik": 0.85,
            "metadata": {"kaynak": "Nöroloji Derneği"}
        },
        {
            "id": "3",
            "icerik": "Diyabet (şeker hastalığı) kan glikozunun yüksek olmasıdır. Tip 1 diyabet pankreasın insülin üretememesi, Tip 2 diyabet insülin direncidir. Belirtileri: çok su içme, sık idrara çıkma, kilo kaybı. Düzenli kan şekeri kontrolü, diyet, egzersiz ve ilaç tedavisi gereklidir.",
            "kategori": "hastalik",
            "guvenilirlik": 0.95,
            "metadata": {"kaynak": "Endokrinoloji Derneği"}
        },
        {
            "id": "4", 
            "icerik": "Kalp krizi (miyokard infarktüsü) kalp kasının oksijensiz kalmasıdır. Belirtiler: göğüs ağrısı, göğüste yanma, nefes darlığı, terleme, bulantı, sol kola yayılan ağrı. ACIL DURUMDUR! Derhal 112'yi arayın. Zaman kritiktir, gecikme kalıcı hasara neden olur.",
            "kategori": "acil",
            "guvenilirlik": 1.0,
            "metadata": {"kaynak": "Kardiyoloji Derneği"}
        },
        {
            "id": "5",
            "icerik": "Vitamin D eksikliği kemik sağlığını olumsuz etkiler. D vitamini güneş ışığıyla sentezlenir, balık, yumurta, süt ürünlerinde bulunur. Eksiklik belirtileri: kemik ağrısı, kas zayıflığı, yorgunluk. Kan tahlili ile 25-OH vitamin D düzeyi ölçülür. Gerekirse D3 vitamini takviyesi alınır.",
            "kategori": "beslenme",
            "guvenilirlik": 0.8,
            "metadata": {"kaynak": "Beslenme Uzmanları"}
        },
        {
            "id": "6",
            "icerik": "Öksürük solunum yollarının irritasyona karşı refleksidir. Kuru öksürük ve balgamlı öksürük olmak üzere ikiye ayrılır. Soğuk algınlığı, grip, bronşit, astım nedeniyle olabilir. Bal, ılık su, buhar inhalasyonu faydalıdır. 2 haftadan uzun süren öksürükte doktora başvurun.",
            "kategori": "semptom",
            "guvenilirlik": 0.85,
            "metadata": {"kaynak": "Göğüs Hastalıkları"}
        },
        {
            "id": "7",
            "icerik": "Hipertansiyon (yüksek tansiyon) kan basıncının 140/90 mmHg üzerinde olmasıdır. Kalp hastalığı, felç riski artırır. Genellikle sessiz seyreder, belirtisiz olabilir. Tuzlu gıda tüketimini azaltın, düzenli egzersiz yapın, kilonuzu kontrol edin. Düzenli ilaç kullanımı gerekebilir.",
            "kategori": "hastalik",
            "guvenilirlik": 0.9,
            "metadata": {"kaynak": "Kardiyoloji Derneği"}
        }
    ]
    
    with open("gelismis_saglik_bilgi_bankasi.json", "w", encoding="utf-8") as f:
        json.dump(kapsamli_tibbi_dokumanlar, f, ensure_ascii=False, indent=2)
    
    return "gelismis_saglik_bilgi_bankasi.json"

# Test sistemi
if __name__ == "__main__":
    print("=== GELISMIS SAGLIK ROCK SISTEMI ===\n")
    
    # Bilgi bankası oluştur ve yükle
    bilgi_dosyasi = gelismis_saglik_bilgi_bankasi_olustur()
    saglik_ai = GelismissSaglikROCK()
    saglik_ai.tibbi_bilgi_yukle(bilgi_dosyasi)
    
    # Test soruları - anlam odaklı
    test_sorulari = [
        "Vücut ısım yüksek ne yapmalıyım?",  # ateş ile eşleşmeli
        "Kafam çok ağrıyor migren mi?",       # baş ağrısı ile eşleşmeli  
        "Şekerim yüksek diyabetik miyim?",    # diyabet ile eşleşmeli
        "Göğsümde sıkışma var kalp mi?",     # kalp krizi ile eşleşmeli
        "D vitamini eksikliği nasıl belli olur?", # vitamin D ile eşleşmeli
        "Sürekli öksürüyorum neden?",        # öksürük ile eşleşmeli
        "Tansiyonum yüksek ne yapmalıyım?"   # hipertansiyon ile eşleşmeli
    ]
    
    print("\n=== AKILLI ARAMA TESTLERI ===")
    
    for soru in test_sorulari:
        print(f"\n{'-'*60}")
        sonuc = saglik_ai.tibbi_soru_cevapla(soru, detayli_log=True)
        
        print(f"[BENZERLIK SKORU] {sonuc['en_yuksek_benzerlik_skoru']}")
        print(f"[BULUNAN DOKUMAN] {sonuc['bulunan_dokuman_sayisi']}")
        print(f"[SISTEM PERFORMANSI] {sonuc['sistem_performansi']}")
        print(f"\n[YANIT OZETI]\n{sonuc['tibbi_yanit'][:300]}...")
    
    print(f"\n=== SISTEM RAPORU ===")
    print(f"Toplam sorgu: {saglik_ai.sorgu_sayisi}")
    print(f"Basarili yanit: {saglik_ai.basarili_sorgu}")
    print(f"Basari orani: {(saglik_ai.basarili_sorgu/saglik_ai.sorgu_sayisi)*100:.1f}%")