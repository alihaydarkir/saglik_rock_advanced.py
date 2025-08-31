# 🫀 CPR Ultra System v4.0

Modüler yapı ile geliştirilmiş CPR eğitim sistemi. Tek odak, doğru yanıtlar, optimize performans.

## 🎯 Özellikler

### ⚡ Ultra Optimizasyon
- **Tek kategori odaklı** arama
- **Akıllı kategori tespiti**
- **Cache sistemi** (100 sorgu)
- **Kesin kelime eşleşmesi** bonusu

### 🧩 Modüler Yapı
- `config.py` - Tüm ayarlar
- `data_processor.py` - Veri işleme
- `query_engine.py` - Sorgu motoru
- `model_core.py` - Ana model
- `ui_main.py` - Kullanıcı arayüzü

### 📊 Doğru Yanıtlar
- **Kategori bazlı filtreleme**
- **Adımlı protokol sunumu**
- **Tek sonuç odaklı** (karmaşa yok)
- **AHA 2020 Guidelines** temel

## 🚀 Kurulum

### 1. Gereksinimler
```bash
pip install -r requirements.txt
```

### 2. Dosya Yapısı
```
cpr_system/
├── main.py                          # 🚀 Ana uygulama
├── config.py                        # 🔧 Sistem ayarları
├── data_processor.py                # 📊 Veri işleme
├── query_engine.py                  # 🔍 Sorgu motoru
├── model_core.py                    # 🤖 Ana model
├── ui_main.py                       # 🎨 UI arayüzü
├── requirements.txt                 # 📦 Gerekli kütüphaneler
├── cpr_egitim_bilgi_bankasi.json   # 📚 CPR veri bankası
└── README.md                        # 📖 Bu dosya
```

### 3. Çalıştırma
```bash
streamlit run main.py
```

## ⚙️ Konfigürasyon

### Model Ayarları (`config.py`)
```python
MODEL_CONFIG = {
    'model_name': 'all-MiniLM-L12-v2',  # 133MB
    'collection_name': 'cpr_ultra_v4',
    'max_tokens': 512
}
```

### Eşik Değerleri
```python
CATEGORY_THRESHOLDS = {
    'aed_protokol': 0.12,
    'cpr_protokol': 0.10,
    'ilac_protokol': 0.08,
    # ...
}
```

## 🎯 Kullanım

### Basit Sorular
```
Epinefrin dozu kaç mg?
AED nasıl kullanılır?
CPR oranı nedir?
```

### Karmaşık Sorular
```
Yetişkinlerde kardiyak arrest durumunda epinefrin dozu ve uygulama şekli nedir?
```

## 📊 Sistem İzleme

### Cache İstatistikleri
- 100 sorgu cache
- Otomatik temizleme
- Hit rate tracking

### Performance Metrics
- Yanıt süreleri
- Başarı oranları
- Kategori doğruluğu

## 🔧 Geliştirme

### Yeni Kategori Ekleme
1. `config.py`'da `CATEGORY_KEYWORDS` güncelle
2. `CATEGORY_THRESHOLDS` eşik ekle
3. JSON'da yeni kategori dokümanları

### Eşik Optimizasyonu
```python
# config.py içinde
CATEGORY_THRESHOLDS['yeni_kategori'] = 0.15
```

### UI Customization
```python
# config.py içinde
UI_CONFIG = {
    'primary_color': '#667eea',
    'secondary_color': '#764ba2'
}
```

## 🧪 Test Modları

### Data Processor Test
```bash
streamlit run data_processor.py
```

### Query Engine Test
```bash
streamlit run query_engine.py
```

### Model Core Test
```bash
streamlit run model_core.py
```

## 🐛 Sorun Giderme

### Common Issues

#### 1. "JSON bulunamadı" Hatası
- `cpr_egitim_bilgi_bankasi.json` dosyasının mevcut olduğundan emin olun
- Dosya yolunu `config.py`'da kontrol edin

#### 2. "Kütüphane eksik" Hatası
```bash
pip install chromadb sentence-transformers streamlit plotly
```

#### 3. "Model yüklenemedi" Hatası
- İnternet bağlantınızı kontrol edin
- Modelin ilk yüklemesi zaman alabilir

#### 4. Düşük Skorlar
- `config.py`'da eşik değerlerini azaltın
- `CATEGORY_KEYWORDS` kelime listelerini genişletin

### Debug Modu
Terminal'de debug bilgileri görüntülenmek için:
```bash
streamlit run main.py --logger.level=debug
```

## 📝 Değişiklik Geçmişi

### v4.0 - Ultra Optimization
- ✅ Modüler yapıya geçiş
- ✅ Tek odak yaklaşımı
- ✅ Kategori bazlı filtreleme
- ✅ Cache sistemi
- ✅ UI okunabilirlik düzeltmeleri

### v3.x - Chunking System
- Query chunking
- Multi-chunk bonus
- Enhanced UI

### v2.x - Professional System
- Büyük model desteği
- Performance tracking
- Advanced metrics

## 📞 Destek

### Acil Durumlar
**112** - Türkiye Acil Servis

### Teknik Destek
Bu sistem eğitim amaçlıdır. Gerçek uygulamalarda mutlaka:
- AHA 2020 Guidelines
- ERC 2021 Standards
- Yerel protokoller

## 🏆 Başarı Metrikleri

- **%85+** başarı oranı hedefi
- **<2s** yanıt süresi
- **Cache hit %30+**
- **Kategori doğruluğu %90+**

---

**🫀 CPR Ultra System v4.0** - Hayat kurtaran bilgi, optimize teknoloji ile buluşuyor.