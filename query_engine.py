# query_engine.py - GELİŞTİRİLMİŞ ARAMA SİSTEMİ v3.0
"""Çoklu embedding, akıllı scoring, gelişmiş kategori tespiti"""

import re
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
from config import get_config

class PowerfulWordExpander:
    """Çok güçlü kelime genişletme sistemi"""
    
    def __init__(self):
        self.word_map = get_config()['word_map']
        self.categories = get_config()['categories']
    
    def multi_expand(self, query: str) -> Tuple[str, str, str]:
        """3 farklı genişletme stratejisi"""
        
        # 1. TEMEL GENİŞLETME - Hızlı
        basic = self._basic_expand(query)
        
        # 2. AKILLI GENİŞLETME - Fuzzy + Context
        smart = self._smart_expand(query)
        
        # 3. DERİN GENİŞLETME - Kategori + Semantic
        deep = self._deep_expand(query)
        
        return basic, smart, deep
    
    def _basic_expand(self, query: str) -> str:
        """Temel genişletme"""
        expanded = query.lower()
        
        # Direkt eş anlamlı ekleme
        for key, synonyms in self.word_map.items():
            if key in expanded:
                expanded += " " + " ".join(synonyms[:2])
        
        return expanded
    
    def _smart_expand(self, query: str) -> str:
        """Akıllı genişletme"""
        expanded = query.lower()
        words = query.lower().split()
        
        # 1. Fuzzy eşleşmeler
        for key, synonyms in self.word_map.items():
            for word in words:
                if len(word) > 3 and len(key) > 3:
                    similarity = SequenceMatcher(None, word, key).ratio()
                    if similarity > 0.75:  # %75+ benzer
                        expanded += f" {key} " + " ".join(synonyms[:2])
        
        # 2. Soru tipi analizi
        if re.search(r'\b(nasıl|ne şekilde)\b', query, re.I):
            expanded += " prosedür yöntem adım protokol teknik"
        
        if re.search(r'\b(nedir|ne|tanım)\b', query, re.I):
            expanded += " açıklama definition bilgi detay"
            
        if re.search(r'\b(kaç|ne kadar|miktar)\b', query, re.I):
            expanded += " doz sayı amount mg milligram"
        
        # 3. Sayısal değer tespiti
        if re.search(r'\d+', query):
            expanded += " doz miktar mg cc ml gram"
        
        return expanded
    
    def _deep_expand(self, query: str) -> str:
        """Derin genişletme"""
        expanded = query.lower()
        words = set(query.lower().split())
        
        # 1. Kategori tespit ve genişletme
        best_category, best_score = self._detect_category(words)
        if best_category:
            category_words = self.categories.get(best_category, [])
            expanded += " " + " ".join(category_words[:3])
        
        # 2. Semantic genişletme - ilişkili kavramlar
        semantic_map = {
            'kalp': ['cardiac', 'miyokard', 'ventrikül', 'atrium'],
            'nefes': ['solunum', 'respiration', 'oksigen', 'ventilation'],
            'çocuk': ['pediatrik', 'infant', 'baby', 'küçük'],
            'acil': ['emergency', 'kritik', 'urgent', 'arrest']
        }
        
        for key, related in semantic_map.items():
            if key in expanded:
                expanded += " " + " ".join(related[:2])
        
        # 3. Bağlam genişletmesi
        if any(word in words for word in ['epinefrin', 'adrenalin', 'ilaç']):
            expanded += " vazopresor mg doz IV intravenöz uygulaması"
        
        if any(word in words for word in ['aed', 'defibrillatör', 'şok']):
            expanded += " elektrot pad joule energy bifazik monofazik"
        
        return expanded
    
    def _detect_category(self, words: set) -> Tuple[str, float]:
        """Basit kategori tespiti"""
        best_cat, best_score = None, 0
        
        for category, keywords in self.categories.items():
            overlap = len(words & set(keywords))
            if overlap > best_score:
                best_score = overlap
                best_cat = category
                
        return best_cat, best_score

class AdvancedCategoryDetector:
    """Gelişmiş kategori tespit sistemi"""
    
    def __init__(self):
        self.categories = get_config()['categories']
        self.word_map = get_config()['word_map']
    
    def analyze_query(self, query: str) -> Dict[str, any]:
        """Tam sorgu analizi"""
        query_lower = query.lower().strip()
        words = set(query_lower.split())
        
        # 1. Kategori skorları
        category_scores = self._calculate_category_scores(query_lower, words)
        
        # 2. En iyi kategori
        best_category = max(category_scores, key=category_scores.get) if category_scores else 'cpr'
        confidence = category_scores.get(best_category, 0) / max(sum(category_scores.values()), 1)
        
        # 3. Sorgu özellikleri
        features = self._extract_features(query_lower)
        
        return {
            'primary_category': best_category,
            'confidence': confidence,
            'all_scores': category_scores,
            'features': features,
            'complexity': self._assess_complexity(query)
        }
    
    def _calculate_category_scores(self, query: str, words: set) -> Dict[str, float]:
        """Kategori skorlarını hesapla"""
        scores = {}
        
        for category, keywords in self.categories.items():
            score = 0
            
            # Tam eşleşmeler - 5 puan
            for keyword in keywords:
                if keyword in query:
                    score += 5
            
            # Kelime kesişimi - 3 puan
            intersection = words & set(keywords)
            score += len(intersection) * 3
            
            # Fuzzy eşleşmeler - 1 puan
            for keyword in keywords:
                for word in words:
                    if len(word) > 3 and len(keyword) > 3:
                        similarity = SequenceMatcher(None, word, keyword).ratio()
                        if similarity > 0.8:
                            score += 1
            
            if score > 0:
                scores[category] = score
        
        return scores
    
    def _extract_features(self, query: str) -> Dict[str, bool]:
        """Sorgu özelliklerini çıkar"""
        return {
            'has_question': bool(re.search(r'\b(nasıl|nedir|ne|kaç|hangi|nerede)\b', query)),
            'has_numbers': bool(re.search(r'\d+', query)),
            'has_dose': bool(re.search(r'\b(mg|doz|miktar|gram)\b', query)),
            'has_procedure': bool(re.search(r'\b(nasıl|adım|prosedür|yöntem)\b', query)),
            'is_pediatric': bool(re.search(r'\b(çocuk|bebek|pediatrik)\b', query)),
            'is_emergency': bool(re.search(r'\b(acil|kritik|arrest|durma)\b', query)),
            'is_detailed': len(query.split()) > 6
        }
    
    def _assess_complexity(self, query: str) -> str:
        """Sorgu karmaşıklığını değerlendir"""
        word_count = len(query.split())
        
        if word_count <= 3:
            return 'basit'
        elif word_count <= 6:
            return 'orta'
        else:
            return 'karmaşık'

class PowerfulSearchEngine:
    """Güçlü arama motoru - çoklu strateji"""
    
    def __init__(self, collection, model):
        self.collection = collection
        self.model = model
        self.config = get_config()
        
        # Güçlü alt sistemler
        self.word_expander = PowerfulWordExpander()
        self.category_detector = AdvancedCategoryDetector()
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'multi_embedding_used': 0,
            'category_accuracy': {},
            'avg_response_time': 0
        }
    
    def powerful_search(self, query: str) -> List[Dict]:
        """Güçlü çoklu arama stratejisi"""
        import time
        start_time = time.time()
        
        self.search_stats['total_searches'] += 1
        
        try:
            print(f"🚀 GÜÇLİ ARAMA BAŞLADI: '{query}'")
            
            # 1. Sorgu analizi
            analysis = self.category_detector.analyze_query(query)
            primary_category = analysis['primary_category']
            confidence = analysis['confidence']
            features = analysis['features']
            
            print(f"📊 ANALİZ: Kategori={primary_category}, Güven={confidence:.2f}")
            print(f"🔍 ÖZELLİKLER: {features}")
            
            # 2. Çoklu genişletme
            basic_exp, smart_exp, deep_exp = self.word_expander.multi_expand(query)
            
            print(f"🔄 GENİŞLETME:")
            print(f"  Basic: {basic_exp[:50]}...")
            print(f"  Smart: {smart_exp[:50]}...")
            print(f"  Deep:  {deep_exp[:50]}...")
            
            # 3. Çoklu embedding arama
            all_results = []
            
            # Her genişletilmiş sorgu için arama
            queries = [
                ("original", query, 1.0),
                ("basic", basic_exp, 0.9),
                ("smart", smart_exp, 1.1), 
                ("deep", deep_exp, 0.8)
            ]
            
            for query_type, query_text, weight in queries:
                results = self._single_search(query_text, primary_category, weight)
                all_results.extend(results)
                print(f"  {query_type}: {len(results)} sonuç")
            
            self.search_stats['multi_embedding_used'] += 1
            
            # 4. Sonuçları birleştir ve optimize et
            final_results = self._merge_and_optimize(all_results, analysis)
            
            # 5. Performance tracking
            response_time = time.time() - start_time
            self.search_stats['avg_response_time'] = (
                self.search_stats['avg_response_time'] + response_time
            ) / 2
            
            print(f"✅ GÜÇLİ ARAMA BİTTİ: {len(final_results)} final sonuç, {response_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            print(f"🚨 GÜÇLİ ARAMA HATASI: {str(e)}")
            return []
    
    def _single_search(self, query_text: str, category: str, weight: float) -> List[Dict]:
        """Tek arama işlemi"""
        try:
            # Embedding oluştur
            embedding = self.model.encode(query_text).tolist()
            
            # ChromaDB'de ara
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=self.config['search']['max_results'],
                include=["documents", "metadatas", "distances"]
            )
            
            # Sonuçları işle
            processed = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i]
                    base_similarity = max(0.0, 1.0 - distance)
                    
                    # Çoklu bonus sistemi
                    bonuses = self._calculate_advanced_bonuses(
                        query_text, 
                        results['documents'][0][i],
                        results['metadatas'][0][i] if results['metadatas'] else {},
                        category
                    )
                    
                    # Final skor
                    final_score = base_similarity * bonuses['total_bonus'] * weight
                    
                    processed.append({
                        'id': results['ids'][0][i] if results['ids'] else f"result_{i}",
                        'icerik': results['documents'][0][i],
                        'skor': final_score,
                        'base_similarity': base_similarity,
                        'bonuses': bonuses,
                        'weight': weight,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'kategori': results['metadatas'][0][i].get('kategori', '') if results['metadatas'] else '',
                        'guvenilirlik': results['metadatas'][0][i].get('guvenilirlik', 0.8) if results['metadatas'] else 0.8
                    })
            
            return processed
            
        except Exception as e:
            print(f"🚨 TEK ARAMA HATASI: {str(e)}")
            return []
    
    def _calculate_advanced_bonuses(self, query: str, document: str, metadata: dict, category: str) -> Dict[str, float]:
        """Gelişmiş bonus hesaplama sistemi"""
        bonuses = {}
        
        # 1. Exact match bonus - geliştirilmiş
        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        doc_words = set(w.lower() for w in document.split() if len(w) > 2)
        
        if query_words:
            exact_matches = len(query_words & doc_words)
            exact_ratio = exact_matches / len(query_words)
            bonuses['exact_match'] = 1.0 + (exact_ratio * 0.4)  # Max %40 bonus
        else:
            bonuses['exact_match'] = 1.0
        
        # 2. Kategori bonus - akıllı
        doc_category = metadata.get('kategori', '')
        if doc_category == category:
            bonuses['category_match'] = 1.3  # %30 bonus
        elif self._are_related_categories(doc_category, category):
            bonuses['category_match'] = 1.1  # %10 bonus
        else:
            bonuses['category_match'] = 1.0
        
        # 3. Güvenilirlik bonus
        reliability = metadata.get('guvenilirlik', 0.8)
        bonuses['reliability'] = 0.8 + (reliability * 0.2)  # 0.8-1.0 arası
        
        # 4. Uzunluk bonus - ideal uzunluk
        doc_length = len(document.split())
        if 20 <= doc_length <= 150:  # İdeal uzunluk
            bonuses['length'] = 1.15
        elif 10 <= doc_length < 20 or 150 < doc_length <= 250:
            bonuses['length'] = 1.05
        else:
            bonuses['length'] = 0.95
        
        # 5. Acillik bonus
        emergency_level = metadata.get('acillik', 'normal')
        if emergency_level == 'kritik':
            bonuses['emergency'] = 1.2
        elif emergency_level == 'yuksek':
            bonuses['emergency'] = 1.1
        else:
            bonuses['emergency'] = 1.0
        
        # 6. Semantic bonus - özel kelimeler
        semantic_bonus = 1.0
        high_value_words = ['epinefrin', 'aed', 'kompresyon', 'defibrilasyon', 'entübasyon']
        for word in high_value_words:
            if word in query.lower() and word in document.lower():
                semantic_bonus += 0.1
        
        bonuses['semantic'] = min(semantic_bonus, 1.5)  # Max %50
        
        # Toplam bonus
        bonuses['total_bonus'] = (
            bonuses['exact_match'] *
            bonuses['category_match'] * 
            bonuses['reliability'] *
            bonuses['length'] *
            bonuses['emergency'] *
            bonuses['semantic']
        )
        
        return bonuses
    
    def _are_related_categories(self, cat1: str, cat2: str) -> bool:
        """İlişkili kategoriler"""
        relations = {
            'cpr': ['çocuk', 'acil'],
            'aed': ['cpr', 'acil'],
            'ilaç': ['cpr', 'acil'],
            'çocuk': ['cpr'],
            'acil': ['cpr', 'aed', 'ilaç']
        }
        
        return cat2 in relations.get(cat1, [])
    
    def _merge_and_optimize(self, all_results: List[Dict], analysis: Dict) -> List[Dict]:
        """Sonuçları birleştir ve optimize et"""
        # Duplicate elimination by ID
        seen_ids = set()
        unique_results = []
        
        # Skora göre sırala
        all_results.sort(key=lambda x: x['skor'], reverse=True)
        
        for result in all_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        # En iyi sonuçları al
        final_results = unique_results[:self.config['search']['max_results']]
        
        # Sonuçları tekrar sırala
        final_results.sort(key=lambda x: x['skor'], reverse=True)
        
        return final_results
    
    def get_search_stats(self) -> Dict:
        """Arama istatistikleri"""
        return self.search_stats.copy()

# ResponseGenerator aynı kalabilir - sadece import değişikliği
class ResponseGenerator:
    """Yanıt oluşturucu - aynı"""
    
    def generate_response(self, query: str, results: List[Dict]) -> str:
        """Ana yanıt oluştur"""
        if not results:
            return self._no_results(query)
        
        best_result = results[0]
        
        response = f"## 🫀 CPR Rehberi v3.0\n\n"
        response += f"**Sorunuz:** {query}\n\n"
        response += f"**Kategori:** {best_result['kategori'].replace('_', ' ').title()}\n\n"
        
        response += "### 📋 Yapılacaklar:\n\n"
        steps = self._split_steps(best_result['icerik'])
        for i, step in enumerate(steps, 1):
            response += f"**{i}.** {step}\n\n"
        
        güvenilirlik = int(best_result['guvenilirlik'] * 100)
        response += f"**📊 Güvenilirlik:** %{güvenilirlik}\n"
        response += f"**🎯 Gelişmiş Skor:** {best_result['skor']:.3f}\n"
        
        # Bonus bilgilerini göster
        if 'bonuses' in best_result:
            bonuses = best_result['bonuses']
            response += f"**🚀 Bonus Detayı:** Exact:{bonuses['exact_match']:.2f}, Kategori:{bonuses['category_match']:.2f}\n"
        
        response += f"**🧠 Model:** Gelişmiş Türkçe v3.0\n\n"
        
        response += "### ⚠️ Hatırlatma\n"
        response += "• **112'yi arayın** acil durumlarda\n"
        response += "• **Bu rehber eğitim amaçlıdır**\n"
        
        return response
    
    def _split_steps(self, content: str) -> List[str]:
        """Adımlara böl"""
        if not content:
            return ["İçerik bulunamadı"]
        
        if re.search(r'\d+\.', content):
            steps = re.split(r'\d+\.', content)[1:]
            clean_steps = [step.strip() for step in steps if step.strip()]
            if clean_steps:
                return clean_steps
        
        if '. ' in content:
            steps = content.split('. ')
            clean_steps = [step.strip() for step in steps if len(step.strip()) > 10]
            if clean_steps:
                return clean_steps
        
        return [content]
    
    def _no_results(self, query: str) -> str:
        """Sonuç yok"""
        response = f"## 🔍 Sonuç Bulunamadı\n\n"
        response += f"**Sorunuz:** {query}\n\n"
        response += "### 💡 Öneriler:\n"
        response += "• Daha basit kelimeler kullanın\n"
        response += "• Ana kelimeleri deneyin (CPR, AED, epinefrin)\n\n"
        
        samples = get_config()['samples']
        response += "### 📝 Örnek Sorular:\n"
        for sample in samples[:4]:
            response += f"• {sample}\n"
        
        return response