# model_core.py - v3.0 SYNTAX HATASI DÜZELTİLDİ
"""Temiz ve çalışan v3.0 sistemi"""

import time
import re
from datetime import datetime
from typing import Dict, List
import streamlit as st

# Imports
from config import get_config
from data_processor import CPRDataProcessor  
from query_engine import PowerfulSearchEngine, ResponseGenerator

# Dependencies
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False

class CPRModelCore:
    """Ana CPR sistem - v3.0"""
    
    def __init__(self):
        self.config = get_config()
        self.data_processor = CPRDataProcessor()
        self.response_generator = ResponseGenerator()
        
        self.chroma_client = None
        self.collection = None
        self.model = None
        self.search_engine = None
        
        self.start_time = datetime.now()
        self.query_count = 0
        self.success_count = 0
        self.response_cache = {}
    
    def start_system(self) -> bool:
        """Sistem başlat v3.0"""
        print("🚀 CPR SİSTEM v3.0 BAŞLATIILIYOR...")
        
        if not CHROMA_OK or not TRANSFORMERS_OK:
            st.error("❌ Kütüphaneler eksik!")
            return False
        
        try:
            # JSON yükle
            if not self.data_processor.json_yukle():
                return False
            
            if not self.data_processor.validate_data():
                return False
            
            # ChromaDB
            if not self._init_chromadb():
                return False
            
            # Model yükle
            if not self._load_model():
                return False
            
            # Database
            if not self._create_database():
                return False
            
            # Güçlü arama sistemi
            self.search_engine = PowerfulSearchEngine(self.collection, self.model)
            
            st.success("✅ CPR v3.0 hazır! (Güçlü Arama)")
            return True
            
        except Exception as e:
            st.error(f"❌ v3.0 Sistem hatası: {str(e)}")
            return False
    
    def _init_chromadb(self) -> bool:
        """ChromaDB başlat"""
        try:
            self.chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            
            collection_name = self.config['model']['collection_name']
            
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                st.info(f"📊 v3.0 Database: {self.collection.count()} doküman")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"version": "v3_powerful"}
                )
                st.info("🆕 v3.0 Database oluşturuldu")
            
            return True
            
        except Exception as e:
            st.error(f"❌ ChromaDB hatası: {str(e)}")
            return False
    
    def _load_model(self) -> bool:
        """Model yükle"""
        try:
            model_name = self.config['model']['model_name']
            
            with st.spinner("🧠 v3.0 Model yükleniyor..."):
                self.model = SentenceTransformer(model_name)
            
            st.success("✅ v3.0 Türkçe model hazır!")
            return True
            
        except Exception as e:
            st.error(f"❌ Model hatası: {str(e)}")
            return False
    
    def _create_database(self) -> bool:
        """Database oluştur"""
        try:
            if self.collection.count() > 0:
                st.info("📊 v3.0 Database hazır")
                return True
            
            st.info("📊 v3.0 Database oluşturuluyor...")
            
            documents = self.data_processor.batch_hazirla()
            
            ids, embeddings, metadatas, contents = [], [], [], []
            
            progress = st.progress(0)
            for i, doc in enumerate(documents):
                ids.append(doc['id'])
                
                embedding = self.model.encode(doc['embedding_icerik']).tolist()
                embeddings.append(embedding)
                
                metadatas.append(doc['metadata'])
                contents.append(doc['icerik'])
                
                progress.progress((i + 1) / len(documents))
            
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=contents,
                ids=ids
            )
            
            st.success(f"✅ v3.0: {len(documents)} doküman eklendi!")
            progress.empty()
            return True
            
        except Exception as e:
            st.error(f"❌ Database hatası: {str(e)}")
            return False
    
    def query(self, question: str) -> Dict:
        """Ana sorgulama v3.0"""
        print(f"\n🚀 v3.0 SORGU: '{question}'")
        
        if not self.search_engine:
            return {"success": False, "response": "❌ Sistem hazır değil!"}
        
        start_time = time.time()
        self.query_count += 1
        
        # Cache
        cache_key = question.strip().lower()
        if cache_key in self.response_cache:
            print("⚡ v3.0 CACHE HIT!")
            cached = self.response_cache[cache_key].copy()
            cached['cache_hit'] = True
            return cached
        
        try:
            # Güçlü arama
            print("🎯 v3.0 Güçlü arama başlıyor...")
            results = self.search_engine.powerful_search(question)
            
            # Eşik kontrolü
            threshold = self.config['search']['default_threshold']
            quality_results = [r for r in results if r['skor'] > threshold]
            
            print(f"📊 v3.0 ANALİZ:")
            print(f"  - Toplam: {len(results)}")
            print(f"  - Eşik: {threshold}")
            print(f"  - Kaliteli: {len(quality_results)}")
            
            # Smart fallback
            if not quality_results and results:
                quality_results = results[:1]
                print("🧠 v3.0 FALLBACK")
            
            # Yanıt oluştur
            if quality_results:
                self.success_count += 1
                print(f"✅ v3.0 BAŞARI: {quality_results[0]['skor']:.3f}")
                response_text = self.response_generator.generate_response(question, quality_results)
                success = True
                best_score = quality_results[0]['skor']
            else:
                print("❌ v3.0 SONUÇ YOK")
                response_text = self.response_generator._no_results(question)
                success = False
                best_score = results[0]['skor'] if results else 0
            
            response_time = time.time() - start_time
            
            result = {
                "success": success,
                "response": response_text,
                "best_score": best_score,
                "total_results": len(results),
                "quality_results": len(quality_results),
                "response_time": response_time,
                "version": "v3.0",
                "cache_hit": False
            }
            
            # Cache kaydet
            if len(self.response_cache) < 100:
                self.response_cache[cache_key] = result.copy()
            
            print(f"📊 v3.0 SONUÇ: {'✅' if success else '❌'}")
            return result
            
        except Exception as e:
            print(f"🚨 v3.0 HATA: {str(e)}")
            return {"success": False, "response": f"❌ v3.0 Hata: {str(e)}"}
    
    def get_stats(self) -> Dict:
        """v3.0 Stats"""
        uptime = datetime.now() - self.start_time
        
        return {
            'system_status': 'v3.0 Aktif' if self.model else 'İnaktif',
            'model_info': 'Türkçe v3.0',
            'document_count': self.collection.count() if self.collection else 0,
            'query_count': self.query_count,
            'success_count': self.success_count,
            'success_rate': f"{(self.success_count/max(1,self.query_count))*100:.1f}%",
            'cache_size': len(self.response_cache),
            'uptime': str(uptime).split('.')[0],
            'version': 'v3.0 Güçlü Sistem'
        }
    
    def clear_cache(self):
        """Cache temizle"""
        self.response_cache.clear()
        st.success("✅ v3.0 Cache temizlendi!")