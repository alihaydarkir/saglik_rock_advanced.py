# ✅ Çalışan CPR Eğitim Sistemi - ChromaDB olmadan
# Bu dosyayı cpr_sistem_calisir.py olarak kaydet

import numpy as np
from typing import List, Dict
import json

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[HATA] Sentence Transformers gerekli: pip install sentence-transformers")


class BasitCPRRetriever:
    """Basit ama çalışır vector arama sistemi"""

    def __init__(self):
        self.dokumanlar = []
        self.embeddings = []
        self.model = None

    def sistem_baslat(self):
        """Model yükleme"""
        if not TRANSFORMERS_AVAILABLE:
            return False

        try:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            return True
        except Exception as e:
            print(f"[HATA] Model yuklenemedi: {str(e)}")
            return False

    def dokumanlar_ekle(self, dokumanlar: List[Dict]):
        """Dökümanları ekle ve embeddingler oluştur"""
        if not self.model:
            print("[HATA] Model yuklenmemis!")
            return False

        print(f"[EMBEDDING] {len(dokumanlar)} dokuman isleniyor...")

        for i, dok in enumerate(dokumanlar):
            self.dokumanlar.append({
                "id": dok.get("id", str(i)),
                "icerik": dok["icerik"],
                "kategori": dok.get("kategori", "genel"),
                "alt_kategori": dok.get("alt_kategori", "genel"),
                "guvenilirlik": dok.get("guvenilirlik", 0.8),
                "acillik_seviyesi": dok.get("acillik_seviyesi", "normal"),
            })

            embedding = self.model.encode(dok["icerik"])
            self.embeddings.append(embedding)

            if (i + 1) % 5 == 0:
                print(f"[PROGRESS] {i + 1}/{len(dokumanlar)} tamamlandi")

        print(f"[BASARI] {len(dokumanlar)} dokuman eklendi!")
        return True

    def arama_yap(self, sorgu: str, top_k: int = 5) -> List[Dict]:
        """Basit cosine similarity arama"""
        if not self.model or not self.dokumanlar:
            return []

        sorgu_embedding = self.model.encode(sorgu)
        sonuclar = []
        for dok, dok_embedding in zip(self.dokumanlar, self.embeddings):
            benzerlik = np.dot(sorgu_embedding, dok_embedding) / (
                np.linalg.norm(sorgu_embedding) * np.linalg.norm(dok_embedding)
            )
            final_skor = benzerlik * dok["guvenilirlik"]

            sonuclar.append({
                "id": dok["id"],
                "icerik": dok["icerik"],
                "kategori": dok["kategori"],
                "alt_kategori": dok["alt_kategori"],
                "benzerlik_skoru": float(benzerlik),
                "final_skor": float(final_skor),
                "guvenilirlik": dok["guvenilirlik"],
                "acillik_seviyesi": dok["acillik_seviyesi"],
            })

        sonuclar.sort(key=lambda x: x["final_skor"], reverse=True)
        return sonuclar[:top_k]


class CPREgitimSistemi:
    """CPR Eğitim sistemi - basit ama çalışır"""

    def __init__(self):
        self.retriever = BasitCPRRetriever()
        self.sorgu_sayisi = 0
        self.basarili_sorgu = 0
        self.cpr_kelimeler = [
            "cpr", "kompresyon", "solunum", "nabız", "resüsitasyon",
            "aed", "defibrilasyon", "şok", "kalp krizi",
            "epinephrin", "amiodarone", "entübasyon",
        ]

    def sistem_baslat(self, veri_dosyasi: str = "cpr_egitim_bilgi_bankasi.json"):
        """Sistem başlatma"""
        print("[BASLANGIC] CPR Egitim Sistemi baslatiliyor...")

        if not self.retriever.sistem_baslat():
            print("[HATA] Retriever baslatilamadi!")
            return False

        try:
            with open(veri_dosyasi, "r", encoding="utf-8") as f:
                veriler = json.load(f)
            print(f"[VERI] {len(veriler)} CPR dokumani yuklendi")

            if not self.retriever.dokumanlar_ekle(veriler):
                print("[HATA] Dokumanlar eklenemedi!")
                return False

            print("[BASARI] CPR Egitim Sistemi hazir!")
            return True
        except FileNotFoundError:
            print(f"[HATA] Veri dosyasi bulunamadi: {veri_dosyasi}")
            return False
        except Exception as e:
            print(f"[HATA] Sistem hatasi: {str(e)}")
            return False

    def soru_sor(self, soru: str) -> Dict:
        """CPR sorusu cevaplama"""
        self.sorgu_sayisi += 1
        print(f"\n[CPR SORU {self.sorgu_sayisi}] {soru}")

        sonuclar = self.retriever.arama_yap(soru, top_k=5)

        # Debug için ilk 3 sonucu yazdır
        for i, s in enumerate(sonuclar[:3]):
            print(f"[DEBUG] {i+1}. {s['kategori']} - Skor: {s['benzerlik_skoru']:.3f}")

        cpr_odakli = any(kelime in soru.lower() for kelime in self.cpr_kelimeler)

        kaliteli_sonuclar = [s for s in sonuclar if s['benzerlik_skoru'] > 0.2]

        if kaliteli_sonuclar:
            self.basarili_sorgu += 1
            yanit = self._cpr_yanit_olustur(soru, kaliteli_sonuclar[:3])
            basarili = True
        else:
            yanit = self._basarisiz_yanit(soru)
            basarili = False

        acil_kelimeler = ['acil', 'kritik', 'kalp durması', 'resüsitasyon']
        acil_durum = any(kelime in soru.lower() for kelime in acil_kelimeler)

        return {
            'soru': soru,
            'yanit': yanit,
            'basarili': basarili,
            'acil_durum': acil_durum,
            'cpr_odakli': cpr_odakli,
            'bulunan_dokuman_sayisi': len(sonuclar),
            'kaliteli_sonuc_sayisi': len(kaliteli_sonuclar),
            'sistem_performansi': f"{self.basarili_sorgu}/{self.sorgu_sayisi}",
            'en_iyi_skor': sonuclar[0]['benzerlik_skoru'] if sonuclar else 0,
            'sonuc_detaylari': [
                (s['benzerlik_skoru'], s['kategori'], s['alt_kategori'])
                for s in kaliteli_sonuclar[:3]
            ]
        }

    def _cpr_yanit_olustur(self, soru: str, sonuclar: List[Dict]) -> str:
        """CPR yanıt oluştur"""
        en_iyi = sonuclar[0]
        baslik = "🚨 KRİTİK CPR PROTOKOLÜ" if en_iyi.get("acillik_seviyesi") == "kritik" else "📋 CPR EĞİTİM PROTOKOLÜ"

        protokol_bloklari = []
        for i, sonuc in enumerate(sonuclar):
            kategori = sonuc.get("alt_kategori") or sonuc.get("kategori")
            protokol_bloklari.append(
                f"[{(kategori or '').upper()}] (Skor: {sonuc['benzerlik_skoru']:.2f})\n{sonuc['icerik']}"
            )

        return f"""{baslik}

SORU: {soru}

=== PROTOKOL BİLGİLERİ ===
{chr(10).join(protokol_bloklari)}

*** NOT ***
Bu protokoller eğitim amaçlıdır. Gerçek uygulamada doktor onayı gerekir.
"""

    def _basarisiz_yanit(self, soru: str) -> str:
        return f"""📋 CPR EĞİTİM SİSTEMİ

SORU: {soru}

SONUÇ: Yeterli bilgi bulunamadı.

ÖNERİLER:
- Daha spesifik sorun (örn: 'AED adımları nedir?')
- Yaş grubunu belirtin (yetişkin/çocuk/bebek)
- İlaç dozu gibi detay verin (örn: 'Epinephrin 1mg')
"""


if __name__ == "__main__":
    sistem = CPREgitimSistemi()
    if sistem.sistem_baslat():
        print("✅ Sistem başarıyla test edildi")
        sonuc = sistem.soru_sor("CPR kompresyon oranı nedir?")
        print(sonuc["yanit"])
    else:
        print("❌ Sistem başlatılamadı")
