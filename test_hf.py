# test_ollama.py - Ollama Yerel AI Testi
import requests
import json

# Ollama'nın çalıştığı adres (varsayılan olarak bu)
OLLAMA_URL = "http://localhost:11434/api/generate"

# Test sorusu
test_question = """Post CPR uygulanan hasta eğer geri döndüyse uygulanacak ilacın adı ve uygulanacak ilacın dozu kilosuna göre nasıl hesaplanmalıdır? İnfizyon olarak mı başlatılır yoksa pushe olarak mı ilaç uygulaması yapılır? Bunun değişimini hastanın entübe olup olmaması mevcut durumu değiştirir mi?"""

print("🧪 Ollama Yerel AI Testi Başlıyor...")
print(f"🔗 Model: llama2")
print(f"❓ Soru: {test_question[:100]}...")
print("-" * 50)

def ask_ollama(question):
    """Ollama'ya soru sor"""
    payload = {
        "model": "llama2",
        "prompt": question,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Ollama hatası: {e}")
        return None

try:
    # Ollama'ya soru gönder
    result = ask_ollama(test_question)
    
    if result and 'response' in result:
        print("✅ BAŞARILI! Yanıt:")
        print("-" * 50)
        print(result['response'])
        print("-" * 50)
    else:
        print("❌ Yanıt alınamadı")
        print("ℹ️ Ollama çalışıyor mu? Terminalde 'ollama serve' komutunu çalıştırmayı deneyin.")
        
except Exception as e:
    print(f"❌ Beklenmeyen hata: {str(e)}")