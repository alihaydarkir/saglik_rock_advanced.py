# main.py - CPR sistemi launcher
"""Basitleştirilmiş CPR Ultra System başlatıcısı"""

import os
import sys
import streamlit as st

# Import path düzeltmesi
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def check_dependencies():
    """Gerekli kütüphaneleri kontrol et"""
    try:
        import chromadb
        import sentence_transformers
        return True, "✅ Kütüphaneler hazır"
    except ImportError as e:
        return False, f"❌ Eksik kütüphane: {str(e)}"

def check_files():
    """Gerekli dosyaları kontrol et"""
    required_files = [
        'config.py',
        'data_processor.py',
        'query_engine.py', 
        'model_core.py',
        'ui_main.py',
        'cpr_egitim_bilgi_bankasi.json'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    return missing

def safe_import():
    """Güvenli UI import"""
    try:
        from ui_main import CPRUserInterface
        return CPRUserInterface(), None
    except Exception as e:
        return None, str(e)

def main():
    """Ana launcher fonksiyonu"""
    st.set_page_config(
        page_title="CPR Ultra System",
        page_icon="🫀",
        layout="wide"
    )
    
    # Başlık
    st.markdown("""
    # 🫀 CPR Ultra System v4.0
    **Temizlenmiş • Basitleştirilmiş • Optimize Edilmiş**
    ---
    """)
    
    # 1. Kütüphane kontrolü
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        st.error(deps_msg)
        st.code("pip install chromadb sentence-transformers streamlit")
        return
    
    st.success(deps_msg)
    
    # 2. Dosya kontrolü
    missing_files = check_files()
    if missing_files:
        st.error("❌ Eksik dosyalar:")
        for file in missing_files:
            st.write(f"• {file}")
        return
    
    st.success("✅ Tüm dosyalar mevcut")
    
    # 3. UI başlat
    ui, error = safe_import()
    if error:
        st.error(f"❌ UI başlatma hatası: {error}")
        
        # Fallback - basit rehber
        st.markdown("---")
        st.markdown("## 🏥 Basit CPR Rehberi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("❤️ Temel CPR", use_container_width=True):
                st.markdown("""
                ### Temel CPR Adımları:
                1. **Yanıtsızlık kontrol** - omuz sarsma
                2. **112'yi ara** - yardım çağır
                3. **Nabız kontrol** - 10 saniye karotis
                4. **30 kompresyon** - 5-6cm derinlik
                5. **2 nefes** - göğüs yükselsin
                6. **Devam et** - yardım gelene kadar
                """)
        
        with col2:
            if st.button("⚡ AED Kullanımı", use_container_width=True):
                st.markdown("""
                ### AED Adımları:
                1. **Cihazı aç** - ses komutlarını dinle
                2. **Elektrotları yapıştır** - göğse
                3. **Analiz** - herkesi uzaklaştır
                4. **Şok** - gerekirse ver
                5. **CPR'a devam** - hemen başla
                """)
        return
    
    # 4. Normal UI çalıştır
    try:
        ui.run()
    except Exception as e:
        st.error(f"❌ UI çalıştırma hatası: {str(e)}")

if __name__ == "__main__":
    main()