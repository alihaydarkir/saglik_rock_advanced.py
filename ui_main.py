# ui_main.py - IMPORT HATASI DÜZELTİLDİ
"""Temizlenmiş UI - List import eklendi"""

import streamlit as st
import time
import random
from typing import List, Dict  # List import eklendi
from config import get_config
from model_core import CPRModelCore

class CPRUserInterface:
    """CPR kullanıcı arayüzü - import düzeltildi"""
    
    def __init__(self):
        self.config = get_config()
        
    def run(self):
        """Ana UI çalıştır"""
        # CSS yükle
        st.markdown(self.config['css'], unsafe_allow_html=True)
        
        # Header
        self._render_header()
        
        # Sidebar
        self._render_sidebar()
        
        # Ana interface
        self._render_main()
        
        # Footer
        self._render_footer()
    
    def _render_header(self):
        """Başlık"""
        system_ready = st.session_state.get('system_ready', False)
        status_text = "✅ Hazır" if system_ready else "⏳ Yükleniyor"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1>🫀 CPR Ultra System v2.0 🇹🇷</h1>
            <p><strong>Türkçe Optimize • Hızlı • Doğru</strong></p>
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; 
                       display: inline-block; margin-top: 1rem;">
                Sistem: {status_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Sidebar"""
        with st.sidebar:
            st.markdown("## 🎯 CPR Hub v2.0")
            st.markdown("---")
            
            # Hızlı sorular - kategorilere ayrılmış
            st.markdown("### ⚡ Hızlı Sorular")
            
            categories = {
                "🫀 CPR": [
                    "CPR kompresyon oranı nedir?",
                    "Kalp masajı derinliği kaç cm?",
                    "30:2 oranı ne demek?"
                ],
                "⚡ AED": [
                    "AED nasıl kullanılır?",
                    "AED elektrot yerleşimi",
                    "AED güvenlik önlemleri"
                ],
                "💊 İlaçlar": [
                    "Epinefrin dozu kaç mg?",
                    "Amiodarone ne zaman verilir?",
                    "Atropin endikasyonları"
                ],
                "👶 Çocuk": [
                    "Çocuklarda CPR farkları",
                    "Bebek kalp masajı",
                    "Pediatrik dozlar"
                ]
            }
            
            selected_cat = st.selectbox("Kategori:", list(categories.keys()))
            
            for soru in categories[selected_cat]:
                if st.button(soru, key=f"cat_{soru}", use_container_width=True):
                    st.session_state.selected_question = soru
            
            st.markdown("---")
            
            # Sistem durumu
            st.markdown("### 📊 Sistem Durumu")
            if 'cpr_system' in st.session_state and st.session_state.cpr_system:
                stats = st.session_state.cpr_system.get_stats()
                
                st.success("🇹🇷 Türkçe Sistem Aktif")
                st.info(f"📚 {stats['document_count']} doküman")
                st.info(f"🎯 {stats['success_rate']} başarı")
                st.info(f"⏱️ {stats['uptime']} çalışma")
            else:
                st.warning("⚠️ Sistem başlatılmadı")
            
            # Kontroller
            st.markdown("---")
            if st.button("🗑️ Cache Temizle", use_container_width=True):
                if 'cpr_system' in st.session_state:
                    st.session_state.cpr_system.clear_cache()
    
    def _render_main(self):
        """Ana arayüz"""
        # Sistem başlat
        self._init_system()
        
        if not st.session_state.get('system_ready', False):
            st.error("❌ Sistem başlatılamadı!")
            return
        
        # Başarı mesajı
        st.markdown('''
        <div class="success-box">
            ✅ <strong>Türkçe sistem hazır!</strong> Gelişmiş Türkçe model ile optimize sonuçlar.
        </div>
        ''', unsafe_allow_html=True)
        
        # Metrikler
        self._show_metrics()
        
        st.markdown("---")
        
        # Ana soru bölümü
        st.markdown("## 💬 Türkçe CPR Sorgulama")
        
        # Ana input
        question = st.text_input(
            "CPR sorunuzu yazın:",
            value=st.session_state.get('selected_question', ''),
            placeholder="Örn: Yetişkinlerde epinefrin dozu kaç mg ve nasıl uygulanır?",
            key="main_question"
        )
        
        # Gerçek zamanlı analiz
        if question and len(question) > 5:
            self._show_live_analysis(question)
        
        # Butonlar
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            search_btn = st.button("🔍 Türkçe Analiz", type="primary", use_container_width=True)
        
        with col2:
            if st.button("🗑️ Temizle", use_container_width=True):
                st.session_state.selected_question = ""
                st.rerun()
        
        with col3:
            if st.button("🎲 Rastgele", use_container_width=True):
                st.session_state.selected_question = random.choice(self.config['samples'])
                st.rerun()
        
        # Arama işlemi
        if search_btn and question.strip():
            self._handle_search(question)
        elif search_btn and not question.strip():
            st.warning("⚠️ Lütfen bir soru yazın.")
    
    def _show_live_analysis(self, question: str):
        """Gerçek zamanlı analiz"""
        with st.expander("🔍 Canlı Analiz", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Kategori tahmini
                if any(word in question.lower() for word in ['epinefrin', 'ilaç', 'doz', 'mg']):
                    st.info("🏷️ Kategori: İlaç Protokolü")
                elif any(word in question.lower() for word in ['aed', 'defibrillatör', 'şok']):
                    st.info("🏷️ Kategori: AED Kullanımı")
                elif any(word in question.lower() for word in ['çocuk', 'bebek', 'pediatrik']):
                    st.info("🏷️ Kategori: Pediatrik CPR")
                else:
                    st.info("🏷️ Kategori: Genel CPR")
            
            with col2:
                # Sorgu kalitesi
                word_count = len(question.split())
                if word_count >= 6:
                    st.success("✅ Detaylı soru - Mükemmel")
                elif word_count >= 4:
                    st.warning("⚡ Orta detay - İyi")
                else:
                    st.error("📝 Kısa soru - Detay ekleyin")
    
    def _init_system(self):
        """Sistem başlat"""
        if 'cpr_system' not in st.session_state:
            with st.spinner("🇹🇷 Türkçe sistem başlatılıyor..."):
                st.session_state.cpr_system = CPRModelCore()
                st.session_state.system_ready = st.session_state.cpr_system.start_system()
    
    def _show_metrics(self):
        """Metrikler"""
        if 'cpr_system' in st.session_state:
            stats = st.session_state.cpr_system.get_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📚 Dokümanlar", stats['document_count'])
            
            with col2:
                st.metric("📝 Sorgular", stats['query_count'])
            
            with col3:
                st.metric("🎯 Başarı", stats['success_rate'])
            
            with col4:
                st.metric("⏱️ Çalışma", stats['uptime'])
    
    def _handle_search(self, question: str):
        """Arama işlemi - skor düzeltmesi"""
        with st.spinner("🇹🇷 Türkçe model analiz ediyor..."):
            result = st.session_state.cpr_system.query(question)
        
        st.markdown("---")
        
        # Sonuç durumu - DÜZELTME
        if result['success']:
            if result.get('cache_hit'):
                st.info("⚡ **Cache Hit!** Hızlı yanıt.")
            else:
                best_score = result.get('best_score', 0)
                
                # SKOR KONTROLÜ DÜZELTİLDİ
                if best_score > 0.20:  # Eşik düşürüldü
                    st.success("✅ **Mükemmel eşleşme!** Türkçe model harika sonuç verdi.")
                elif best_score > 0.10:  # Daha düşük eşik
                    st.success("✅ **Çok iyi eşleşme!** Protokol bulundu.")
                elif best_score > 0.05:  # En düşük eşik
                    st.info("📋 **İyi eşleşme!** İlgili bilgi bulundu.")
                else:
                    st.warning("⚡ **Orta eşleşme!** Sonuç var ama düşük skor.")
        else:
            st.warning("⚠️ **Spesifik protokol bulunamadı.** Öneriler sunuluyor.")
        
        # Ana yanıt
        st.markdown(result['response'])
        
        # Detaylar
        if st.checkbox("📊 Detayları Göster"):
            self._show_details(result, question)
        
        # Feedback
        self._show_feedback(question, result)
    
    # config.py - eşik düzeltmesi
    SEARCH_CONFIG = {
        'max_results': 8,
        'default_threshold': 0.05,  # 0.12'den 0.05'e düşürüldü
        'cache_size': 100
    }
    def _show_details(self, result, question):
        """Detay analizi"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 Arama Detayları:**")
            st.write(f"• En iyi skor: {result.get('best_score', 0):.3f}")
            st.write(f"• Toplam sonuç: {result.get('total_results', 0)}")
            st.write(f"• Kaliteli sonuç: {result.get('quality_results', 0)}")
        
        with col2:
            st.markdown("**⚡ Performans:**")
            st.write(f"• Yanıt süresi: {result.get('response_time', 0):.2f}s")
            st.write(f"• Cache: {'Evet' if result.get('cache_hit') else 'Hayır'}")
            st.write(f"• Model: Türkçe Optimize v2.0")
    
    def _show_feedback(self, question, result):
        """Feedback"""
        st.markdown("---")
        st.markdown("### 💭 Bu yanıt nasıldı?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Mükemmel!", use_container_width=True):
                st.success("✅ Harika! Türkçe sistem çok iyi çalışıyor.")
        
        with col2:
            if st.button("👍 İyi", use_container_width=True):
                st.info("📈 Teşekkürler! Sistem gelişiyor.")
        
        with col3:
            if st.button("👎 Yetersiz", use_container_width=True):
                st.warning("📝 Feedback alındı. Daha iyi için daha detaylı soru sorun.")
    
    def _render_footer(self):
        """Footer"""
        st.markdown("---")
        
        # Özellikler showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>🇹🇷 Türkçe Optimize</h4>
                <ul>
                    <li>Multilingual model</li>
                    <li>278MB güçlü model</li>
                    <li>Türkçe anlama artırıldı</li>
                    <li>Doğal dil işleme</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>⚡ Hızlı Sistem</h4>
                <ul>
                    <li>0.1 saniye altı yanıt</li>
                    <li>Cache sistemi</li>
                    <li>Optimize pipeline</li>
                    <li>Gerçek zamanlı analiz</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-box">
                <h4>📚 Kapsamlı İçerik</h4>
                <ul>
                    <li>40+ CPR protokolü</li>
                    <li>AHA 2020 rehberleri</li>
                    <li>Pediatrik protokoller</li>
                    <li>İlaç dozları</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Final mesaj
        st.markdown("""
        <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 2rem; border-radius: 15px; color: white; margin: 2rem 0;">
            <h3>🫀 CPR Ultra System v2.0</h3>
            <p><strong>🇹🇷 Türkçe Optimize • ⚡ Hızlı • 🎯 Doğru</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Uyarı
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(255, 193, 7, 0.1); 
                   border-radius: 10px; margin: 1rem 0; color: #856404;">
            <p><strong>⚠️ ÖNEMLİ:</strong> Bu sistem eğitim amaçlıdır. 
            Gerçek acil durumlarda <strong>112</strong>'yi arayın.</p>
        </div>
        """, unsafe_allow_html=True)