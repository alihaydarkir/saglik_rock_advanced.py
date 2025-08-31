# data_processor.py - Veri işleme
"""CPR JSON verilerini yükler ve hazırlar"""

import json
import streamlit as st
from datetime import datetime
from config import get_config

class CPRDataProcessor:
    """CPR veri işleme sınıfı - basitleştirildi"""
    
    def __init__(self):
        self.config = get_config()
        self.bilgi_bankasi = []
        
    def json_yukle(self) -> bool:
        """JSON dosyasını UTF-8 ile yükle"""
        try:
            # UTF-8 encoding ile yükle - Türkçe karakter sorunu çözülsün
            with open('cpr_egitim_bilgi_bankasi.json', 'r', encoding='utf-8') as f:
                self.bilgi_bankasi = json.load(f)
            
            st.success(f"✅ {len(self.bilgi_bankasi)} doküman yüklendi")
            return True
            
        except FileNotFoundError:
            st.error("❌ JSON dosyası bulunamadı!")
            return False
        except Exception as e:
            st.error(f"❌ Yükleme hatası: {str(e)}")
            return False
    
    def dokuman_hazirla(self, dok: dict) -> dict:
        """Tek doküman hazırla - embedding için"""
        # Temel içerik temizleme
        icerik = dok['icerik']
        kategori = dok.get('kategori', '')
        
        # Embedding için içerik - basit birleştirme
        embedding_icerik = f"{icerik} {kategori}"
        
        # Metadata hazırla
        metadata = {
            'kategori': kategori,
            'guvenilirlik': float(dok.get('guvenilirlik', 0.8)),
            'acillik': dok.get('acillik_seviyesi', 'normal'),
            'kaynak': dok.get('metadata', {}).get('kaynak', 'AHA Guidelines'),
            'processed_at': datetime.now().isoformat()
        }
        
        return {
            'id': dok.get('id', f"doc_{len(self.bilgi_bankasi)}"),
            'icerik': icerik,
            'embedding_icerik': embedding_icerik,
            'metadata': metadata
        }
    
    def batch_hazirla(self) -> list:
        """Tüm dokümanları hazırla"""
        hazir_dokumanlar = []
        
        for dok in self.bilgi_bankasi:
            hazir_dok = self.dokuman_hazirla(dok)
            hazir_dokumanlar.append(hazir_dok)
        
        return hazir_dokumanlar
    
    def validate_data(self) -> bool:
        """Veri doğrulama - basit kontrol"""
        if not self.bilgi_bankasi:
            st.error("❌ Veri yok!")
            return False
        
        # İlk dokümanı kontrol et
        first_doc = self.bilgi_bankasi[0]
        required_fields = ['id', 'icerik', 'kategori']
        
        for field in required_fields:
            if field not in first_doc:
                st.error(f"❌ Eksik alan: {field}")
                return False
        
        st.success("✅ Veri geçerli!")
        return True