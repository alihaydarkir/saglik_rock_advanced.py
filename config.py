# config.py - v3.0 GELİŞTİRİLMİŞ AYARLAR
"""v3.0 için optimize edilmiş konfigürasyon"""

# Model ayarları - v3.0 için optimize
MODEL_CONFIG = {
    'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # 🇹🇷 TÜRKÇE OPTİMİZE
    'collection_name': 'cpr_ultra_v3_powerful',  # v3.0 collection
    'max_tokens': 512,
    'description': '278MB Türkçe optimize + v3.0 Güçlü Arama'
}

# UI ayarları - v3.0
UI_CONFIG = {
    'page_title': 'CPR Ultra System v3.0 🚀',  # v3.0 versiyonu
    'page_icon': '🫀',
    'layout': 'wide'
}

# Arama ayarları - v3.0 için optimize
SEARCH_CONFIG = {
    'max_results': 10,  # 8'den 10'a artırıldı - çoklu embedding için
    'default_threshold': 0.08,  # v3.0 için optimize edildi
    'cache_size': 200,  # Cache artırıldı - güçlü sistem için
    'multi_embedding_enabled': True,  # YENİ: Çoklu embedding aktif
    'advanced_bonuses': True,  # YENİ: Gelişmiş bonus sistemi
    'fuzzy_matching': True  # YENİ: Fuzzy matching aktif
}

# Kategori anahtar kelimeleri - Türkçe odaklı genişletildi
CATEGORY_KEYWORDS = {
    'cpr': ['cpr', 'kalp masajı', 'kompresyon', 'canlandırma', 'resüsitasyon', 'temel yaşam desteği', 'kardiyopulmoner'],
    'aed': ['aed', 'defibrillatör', 'şok', 'elektrot', 'otomatik defibrillatör', 'defibrilasyon', 'elektrik şoku'],
    'ilaç': ['epinefrin', 'adrenalin', 'amiodarone', 'atropin', 'lidokain', 'vazopresor', 'antiaritmik', 'doz', 'mg'],
    'hava_yolu': ['entübasyon', 'oksijen', 'solunum', 'ventilasyon', 'airway', 'bag mask', 'laryngoscope'],
    'çocuk': ['bebek', 'çocuk', 'pediatrik', 'infant', 'yenidoğan', 'küçük', 'child', 'baby']
}

# Kelime genişletme - Türkçe için optimize
WORD_MAP = {
    'epinefrin': ['adrenalin', 'vazopresor', 'epinephrine', 'vazopresör'],
    'cpr': ['kalp masajı', 'canlandırma', 'resüsitasyon', 'kardiyopulmoner', 'temel yaşam desteği', 'canlandırma'],
    'aed': ['defibrillatör', 'şok cihazı', 'otomatik defibrillatör', 'elektrik şoku', 'defibrilasyon'],
    'kompresyon': ['basınç', 'göğüs basısı', 'masaj', 'basma', 'kompresif', 'sıkıştırma'],
    'çocuk': ['pediatrik', 'infant', 'bebek', 'küçük', 'child', 'yenidoğan'],
    'doz': ['dozu', 'miktar', 'mg', 'milligram', 'amount', 'dose', 'dozaj'],
    'oran': ['ratio', 'rate', 'hız', 'frekans', 'proporsiyon', 'nisbet'],
    'derinlik': ['depth', 'cm', 'santimetre', 'mesafe', 'uzunluk'],
    'amiodarone': ['amiodaron', 'antiaritmik', 'kordarone', 'ritim düzenleyici'],
    'atropin': ['atropine', 'antikolinerjik', 'bradikardi ilacı'],
    'nasıl': ['how', 'ne şekilde', 'hangi yöntem', 'prosedür', 'adımlar'],
    'nedir': ['what', 'ne', 'tanım', 'definition', 'açıklama'],
    'kaç': ['how much', 'how many', 'ne kadar', 'miktar', 'quantity', 'sayı']
}

# Örnek sorular - Türkçe optimize
SAMPLE_QUESTIONS = [
    "Epinefrin dozu kaç mg ve nasıl uygulanır?",
    "AED nasıl kullanılır adım adım?",
    "CPR kompresyon oranı ve derinliği nedir?", 
    "Çocuklarda CPR nasıl farklıdır?",
    "Amiodarone dozu ve endikasyonları neler?",
    "Entübasyon ne zaman gereklidir?",
    "Kalp durmasında ilk yapılacaklar",
    "Hipotermik arrest protokolü nedir?"
]

# CSS - hafif güncellenmiş
CSS_STYLES = """
<style>
.success-box { 
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
    padding: 1.2rem; border-radius: 10px; color: #155724; margin: 1rem 0; 
    border-left: 4px solid #28a745; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.warning-box { 
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
    padding: 1.2rem; border-radius: 10px; color: #856404; margin: 1rem 0;
    border-left: 4px solid #ffc107; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.info-box { 
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
    padding: 1.2rem; border-radius: 10px; color: #0d47a1; margin: 1rem 0;
    border-left: 4px solid #2196f3; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
"""

def get_config():
    """Türkçe optimize konfigürasyon"""
    return {
        'model': MODEL_CONFIG,
        'ui': UI_CONFIG,
        'search': SEARCH_CONFIG,
        'categories': CATEGORY_KEYWORDS,
        'word_map': WORD_MAP,
        'samples': SAMPLE_QUESTIONS,
        'css': CSS_STYLES
    }