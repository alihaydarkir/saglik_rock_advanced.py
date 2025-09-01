# test_ollama_with_translate_v2.py - PARÇALI ÇEVİRİ
import requests
import json
import urllib.parse
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
TRANSLATE_URL = "https://api.mymemory.translated.net/get"

# İNGİLİZCE PROMPT - DAHA KISA
test_question = """
As an emergency medicine specialist, answer concisely:

For adult ROSC after cardiac arrest:
1. First medications used?
2. Dose calculation methods? 
3. Infusion vs push?
4. Intubation effects?

Short professional response:
"""

print("🧪 Ollama + Parçalı Çeviri Testi...")
print("-" * 50)

def ask_ollama(prompt, model_name="gemma:2b", timeout=30):
    """Ollama'dan İngilizce yanıt al"""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 300  # DAHA KISA YANIT
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Ollama hatası: {e}")
        return None

def translate_to_turkish(text):
    """İngilizce'den Türkçe'ye çeviri (500 karakter sınırı için parçalı)"""
    try:
        # Metni 500 karakterlik parçalara böl
        chunks = [text[i:i+400] for i in range(0, len(text), 400)]
        translated_chunks = []
        
        for chunk in chunks:
            text_encoded = urllib.parse.quote(chunk)
            response = requests.get(f"{TRANSLATE_URL}?q={text_encoded}&langpair=en|tr", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                translated_chunks.append(data['responseData']['translatedText'])
            else:
                translated_chunks.append(chunk)  # Çeviri başarısızsa orijinal
        
        return " ".join(translated_chunks)
            
    except Exception as e:
        print(f"❌ Çeviri hatası: {e}")
        return text

try:
    # 1. ÖNCE OLLAMA'DAN İNGİLİZCE YANIT AL
    print("🤖 Ollama'dan İngilizce yanıt alınıyor...")
    result = ask_ollama(test_question, "gemma:2b")
    
    if result and 'response' in result:
        english_response = result['response']
        
        print("✅ İNGİLİZCE YANIT ALINDI!")
        print(f"📏 Uzunluk: {len(english_response)} karakter")
        
        # 2. TÜRKÇE'YE ÇEVİR (PARÇALI)
        print("🌍 Türkçe'ye çevriliyor (parçalı)...")
        turkish_response = translate_to_turkish(english_response)
        
        print("✅ TÜRKÇE YANIT:")
        print("-" * 60)
        print(turkish_response)
        print("-" * 60)
        
        # 3. DOSYAYA KAYDET
        with open("turkce_sonuc_parcali.txt", "w", encoding="utf-8") as f:
            f.write("=== KISA SORU ===\n")
            f.write(test_question)
            f.write("\n\n=== İNGİLİZCE YANIT ===\n")
            f.write(english_response)
            f.write("\n\n=== TÜRKÇE ÇEVİRİ ===\n")
            f.write(turkish_response)
        
        print("💾 Sonuçlar 'turkce_sonuc_parcali.txt' dosyasına kaydedildi")
        
    else:
        print("❌ Yanıt alınamadı")
        
except Exception as e:
    print(f"❌ Genel hata: {str(e)}")