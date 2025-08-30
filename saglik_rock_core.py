# Ana Sağlık ROCK Sistemi - Core Engine
# Bu dosyayı saglik_rock_core.py olarak kaydet

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[UYARI] Sentence Transformers bulunamadi. Kurulum: pip install sentence-transformers")

@dataclass
class TibbiDokuman:
    """Tıbbi döküman veri yapısı"""
    id: str
    icerik: str
    tibbi_kategori: str
    guvenilirlik_skoru: float
    semantic_embedding: Optional[np.ndarray] = None
    metadata: Dict = None

class SaglikRetriever:
    """Sağlık alanı için akıllı retrieval sistemi"""
    
    def __init__(self):
        self.dokumanlar: List[TibbiDokuman] = []
        self.model = None
        
    def model_yukle(self):
        """Sentence Transformer modelini yükle"""
        if TRANSFORMERS_AVAILABLE and not self.model:
            print("[YUKLENIYOR] AI modeli yukleniyor...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("[BASARI] AI modeli yuklendi!")
        return self.model is not None
    
    def dokumanlar_ekle(self, docs: List[Dict]):
        """Dökümanları ekle ve embeddingler oluştur"""
        if not self.model:
            self.model_yukle()
            
        print(f"[ISLEME] {len(docs)} dokuman ekleniyor...")
        
        for i, doc in enumerate(docs):
            yeni_dok = TibbiDokuman(
                id=doc.get('id', str(i)),
                icerik=doc['icerik'],
                tibbi_kategori=doc.get('kategori', 'genel'),
                guvenilirlik_skoru=doc.get('guvenilirlik', 0.8),
                metadata=doc.get('metadata', {})
            )
            
            if self.model:
                yeni_dok.semantic_embedding = self.model.encode(yeni_dok.icerik)
                print(f"[EMBEDDING] Dokuman {i+1}/{len(docs)} islendi")
            
            self.dokumanlar.append(yeni_dok)
        
        print(f"[TAMAMLANDI] {len(docs)} dokuman eklendi!")
    
    def arama_yap(self, sorgu: str, top_k: int = 5) -> List[Tuple[float, TibbiDokuman]]:
        """Semantik arama yap"""
        if not self.dokumanlar:
            return []
        
        if not self.model:
            return []
        
        # Sorgu embedding'i
        sorgu_embedding = self.model.encode(sorgu)
        
        # Benzerlik hesapla
        skorlar = []
        for dokuman in self.dokumanlar:
            if dokuman.semantic_embedding is not None:
                benzerlik = np.dot(sorgu_embedding, dokuman.semantic_embedding)
                final_skor = benzerlik * dokuman.guvenilirlik_skoru
                skorlar.append((final_skor, dokuman))
        
        # Sırala ve top-k döndür
        skorlar.sort(reverse=True, key=lambda x: x[0])
        return skorlar[:top_k]

class SaglikKontrol:
    """Sağlık için güvenlik kontrol sistemi"""
    
    def __init__(self):
        self.acil_kelimeler = ['acil', 'kalp krizi', 'göğüs ağrısı', 'felç', 'zehirlenme', 'kan', 'koma']
        self.kategori_bonus = {
            'tansiyon': ['hipertansiyon', 'kan basıncı', 'yüksek tansiyon'],
            'şeker': ['diyabet', 'glikoz', 'kan şekeri'], 
            'kalp': ['kalp krizi', 'miyokard', 'kardiyak'],
            'baş': ['migren', 'kafa', 'baş ağrısı'],
            'ateş': ['vücut ısısı', 'sıcaklık', 'yüksek ateş']
        }
    
    def kontrol_ve_filtrele(self, skorlu_sonuclar: List[Tuple[float, TibbiDokuman]], 
                           sorgu: str) -> Tuple[str, bool, List[Tuple[float, TibbiDokuman]]]:
        """İçerik kontrol et, filtrele ve acil durum tespit et"""
        
        if not skorlu_sonuclar:
            return "İlgili bilgi bulunamadı.", False, []
        
        # Acil durum kontrolü
        acil_durum = any(acil in sorgu.lower() for acil in self.acil_kelimeler)
        
        # Kategori bonusu uygula
        sorgu_lower = sorgu.lower()
        bonus_uygulandi = False
        
        for terim, bonus_listesi in self.kategori_bonus.items():
            if terim in sorgu_lower:
                yeni_skorlar = []
                for skor, dok in skorlu_sonuclar:
                    bonus_var = any(bonus_term in dok.icerik.lower() for bonus_term in bonus_listesi)
                    if bonus_var:
                        yeni_skor = skor * 1.5  # %50 bonus
                        yeni_skorlar.append((yeni_skor, dok))
                        bonus_uygulandi = True
                        print(f"[BONUS] '{terim}' icin {dok.tibbi_kategori} dokuman bonusu aldi")
                    else:
                        yeni_skorlar.append((skor, dok))
                
                if bonus_uygulandi:
                    skorlu_sonuclar = sorted(yeni_skorlar, reverse=True, key=lambda x: x[0])
                break
        
        # İçerik oluştur
        icerik = self._icerik_olustur(skorlu_sonuclar[:3])
        
        return icerik, acil_durum, skorlu_sonuclar[:3]
    
    def _icerik_olustur(self, sonuclar: List[Tuple[float, TibbiDokuman]]) -> str:
        """Filtrelenmiş içerik oluştur"""
        icerik_parcalari = []
        
        for skor, dok in sonuclar:
            if skor >= 2.5:  # Minimum threshold
                kisaltilmis = (dok.icerik[:400] + "..." 
                              if len(dok.icerik) > 400 
                              else dok.icerik)
                
                icerik_parcalari.append(
                    f"[{dok.tibbi_kategori.upper()}] (Güvenilirlik: {dok.guvenilirlik_skoru}/1.0)\n"
                    f"{kisaltilmis}"
                )
        
        return "\n\n---\n\n".join(icerik_parcalari) if icerik_parcalari else "Yeterli güvenilirlikte bilgi bulunamadı."

class SaglikROCKSistemi:
    """Ana Sağlık ROCK sistemi"""
    
    def __init__(self):
        self.retriever = SaglikRetriever()
        self.kontrol = SaglikKontrol()
        self.sorgu_sayisi = 0
        self.basarili_sorgu = 0
    
    def sistem_baslat(self, bilgi_bankasi: List[Dict] = None):
        """Sistemi başlat ve bilgi bankasını yükle"""
        # Model yükle
        model_yuklendi = self.retriever.model_yukle()
        
        if not model_yuklendi:
            print("[HATA] Model yuklenemedi!")
            return False
        
        # Varsayılan bilgi bankası
        if not bilgi_bankasi:
            bilgi_bankasi = self._varsayilan_bilgi_bankasi()
        
        # Dökümanları yükle
        self.retriever.dokumanlar_ekle(bilgi_bankasi)
        
        print("[BASARI] Sistem hazir!")
        return True
    
    def soru_sor(self, soru: str) -> Dict:
        """Ana ROCK pipeline"""
        self.sorgu_sayisi += 1
        
        print(f"\n[SORGU {self.sorgu_sayisi}] {soru}")
        
        # 1. Retrieve - Arama yap
        sonuclar = self.retriever.arama_yap(soru, top_k=5)
        
        # 2. Control - Kontrol ve filtrele
        icerik, acil_durum, final_sonuclar = self.kontrol.kontrol_ve_filtrele(sonuclar, soru)
        
        # 3. Generate - Yanıt oluştur
        if final_sonuclar and "bulunamadı" not in icerik:
            self.basarili_sorgu += 1
            yanit = self._yanit_olustur(soru, icerik, acil_durum)
            basarili = True
        else:
            yanit = self._basarisiz_yanit()
            basarili = False
        
        return {
            "soru": soru,
            "yanit": yanit,
            "icerik": icerik,
            "acil_durum": acil_durum,
            "basarili": basarili,
            "bulunan_dokuman_sayisi": len(sonuclar),
            "sistem_performansi": f"{self.basarili_sorgu}/{self.sorgu_sayisi}",
            "sonuc_detaylari": [(s, d.tibbi_kategori, d.id) for s, d in final_sonuclar] if final_sonuclar else []
        }
    
    def _yanit_olustur(self, soru: str, icerik: str, acil: bool) -> str:
        """Profesyonel yanıt oluştur"""
        if acil:
            acil_uyarisi = "🚨 ACIL DURUM UYARISI: Bu ciddi bir durum olabilir! Derhal 112'yi arayın!\n\n"
        else:
            acil_uyarisi = ""
        
        return f"""{acil_uyarisi}=== SAĞLIK AI ASİSTANI ===

SORU: {soru}

BİLGİ BANKASI ANALİZİ:
{icerik}

*** ÖNEMLİ TIBBİ UYARI ***
• Bu bilgiler genel bilgilendirme amaçlıdır
• Kesin tanı için mutlaka doktor kontrolü gerekir  
• İlaç kullanımı öncesi hekim onayı alın
• Acil durumlarda 112'yi arayın

Sağlıklı günler dileriz! 🏥"""
    
    def _basarisiz_yanit(self) -> str:
        """Başarısız sorgu yanıtı"""
        return """=== SAĞLIK AI ASİSTANI ===

SONUÇ: İlgili konuda güvenilir bilgi bulunamadı.

ÖNERİLER:
• Sorunuzu farklı kelimelerle ifade edin
• Daha spesifik sorular sorun
• Mutlaka bir sağlık uzmanı ile görüşün

ACIL DURUMLAR: 112

Sağlıklı günler! 🏥"""
    
    def _varsayilan_bilgi_bankasi(self) -> List[Dict]:
        """Genişletilmiş bilgi bankasını yükle"""
        try:
            # Önce genişletilmiş veri tabanını dene
            with open("genisletilmis_saglik_bilgi_bankasi.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"[BASARI] Genişletilmiş veri tabanı yüklendi: {len(data)} döküman")
                return data
        except FileNotFoundError:
            print("[UYARI] Genişletilmiş veri bulunamadı, varsayılan veri kullanılıyor...")
            # Varsayılan küçük veri seti
            return [
                {
                    "id": "1",
                    "icerik": "Ateş vücut sıcaklığının normalin üzerine çıkmasıdır. 37.5°C üzeri ateş sayılır. Yüksek ateş, enfeksiyon belirtisi olabilir. Bol sıvı alın, dinlenin. 38.5°C üzeri ateşte doktora başvurun. Parasetamol veya ibuprofen ateş düşürücü olarak kullanılabilir.",
                    "kategori": "semptom",
                    "guvenilirlik": 0.9,
                    "metadata": {"kaynak": "Sağlık Bakanlığı"}
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
                    "icerik": "Hipertansiyon (yüksek tansiyon) kan basıncının 140/90 mmHg üzerinde olmasıdır. Kalp hastalığı, felç riski artırır. Genellikle sessiz seyreder, belirtisiz olabilir. Tuzlu gıda tüketimini azaltın, düzenli egzersiz yapın, kilonuzu kontrol edin. Düzenli ilaç kullanımı gerekebilir.",
                    "kategori": "hastalik",
                    "guvenilirlik": 0.9,
                    "metadata": {"kaynak": "Kardiyoloji Derneği"}
                }
            ]

# Test fonksiyonu
if __name__ == "__main__":
    print("=== SAGLIK ROCK SISTEMI TEST ===")
    
    # Sistemi başlat
    saglik_ai = SaglikROCKSistemi()
    if saglik_ai.sistem_baslat():
        
        # Test soruları
        test_sorulari = [
            "Ateşim var ne yapmalıyım?",
            "Kafam ağrıyor migren mi?", 
            "Şekerim yüksek diyabet mi?",
            "Tansiyonum yüksek ne yapmalıyım?"
        ]
        
        for soru in test_sorulari:
            sonuc = saglik_ai.soru_sor(soru)
            print(f"\n{'='*60}")
            print(f"[BASARILI: {'EVET' if sonuc['basarili'] else 'HAYIR'}]")
            print(f"[ACIL: {'EVET' if sonuc['acil_durum'] else 'HAYIR'}]")
            print(f"[DOKUMAN: {sonuc['bulunan_dokuman_sayisi']}]")
            print(f"\n{sonuc['yanit'][:300]}...")
        
        print(f"\n=== PERFORMANS RAPORU ===")
        print(f"Sistem performansı: {saglik_ai.basarili_sorgu}/{saglik_ai.sorgu_sayisi}")
        print(f"Başarı oranı: {(saglik_ai.basarili_sorgu/saglik_ai.sorgu_sayisi)*100:.1f}%")
    else:
        print("[HATA] Sistem başlatılamadı!")