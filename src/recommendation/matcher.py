"""
Motor de matching y cálculo de relevancia
"""
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class NewsMatcher:
    """Motor de matching de noticias con perfiles de usuario"""
    
    def __init__(self):
        """Inicializa el matcher"""
        pass
    
    def calculate_relevance(
        self,
        user_vector: np.ndarray,
        article_vector: np.ndarray,
        article_categories: List[str],
        user_categories: List[str],
        article_sentiment: Optional[Dict] = None,
        user_sentiment_preference: Optional[str] = None,
        article_section: Optional[str] = None,
        user_section_preference: Optional[List[str]] = None,
        article_date: Optional[str] = None
    ) -> float:
        """
        Calcula la relevancia de un artículo para un usuario
        
        Args:
            user_vector: Vector del perfil del usuario
            article_vector: Vector del artículo
            article_categories: Categorías detectadas en el artículo
            user_categories: Categorías de interés del usuario
            article_sentiment: Sentimiento del artículo
            user_sentiment_preference: Preferencia de sentimiento del usuario
            article_section: Sección del artículo
            user_section_preference: Secciones preferidas del usuario
            article_date: Fecha del artículo (formato ISO 8601)
            
        Returns:
            Score de relevancia (0-1)
        """
        # Similitud coseno base
        if user_vector.shape != article_vector.shape:
            logger.warning("Vectores de dimensiones diferentes, ajustando...")
            min_dim = min(len(user_vector), len(article_vector))
            user_vector = user_vector[:min_dim]
            article_vector = article_vector[:min_dim]
        
        cosine_sim = cosine_similarity([user_vector], [article_vector])[0][0]
        
        # Normalizar a 0-1
        base_score = (cosine_sim + 1) / 2
        
        # Ponderar por categorías coincidentes
        category_bonus = 0.0
        if user_categories and article_categories:
            common_categories = set(user_categories) & set(article_categories)
            if common_categories:
                category_bonus = len(common_categories) / max(len(user_categories), len(article_categories))
        
        # Ponderar por sentimiento (si hay preferencia)
        sentiment_bonus = 0.0
        if user_sentiment_preference and article_sentiment:
            article_label = article_sentiment.get('label', 'neutral')
            if article_label == user_sentiment_preference:
                sentiment_bonus = 0.1
        
        # Ponderar por sección (si hay preferencia)
        section_bonus = 0.0
        if user_section_preference and article_section:
            if article_section in user_section_preference:
                section_bonus = 0.1
        
        
        
        # Calcular score final con factor de recencia
        # El factor de recencia multiplica el score base para dar más peso a artículos recientes
        final_score = base_score * 0.5 + category_bonus * 0.3 + sentiment_bonus + section_bonus
        
        # Asegurar que esté en rango 0-1
        final_score = min(1.0, max(0.0, final_score))
        
        return final_score
    
    def match_articles(
        self,
        user_profile: Dict,
        articles: List[Dict],
        top_k: int = 10
    ) -> List[Tuple[Dict, float, Dict]]:
        """
        Encuentra los artículos más relevantes para un usuario
        
        Args:
            user_profile: Perfil del usuario
            articles: Lista de artículos con sus vectores y metadatos
            top_k: Número de artículos a retornar
            
        Returns:
            Lista de tuplas (artículo, score, justificación)
        """
        user_vector = np.array(user_profile['vector'])
        user_categories = user_profile.get('categories', [])
        
        scored_articles = []
        
        for article in articles:
            article_vector = np.array(article.get('vector', []))
            article_categories = article.get('categories', [])
            article_sentiment = article.get('sentiment')
            article_section = article.get('section')
            article_date = article.get('source_metadata', {}).get('date')
            
            score = self.calculate_relevance(
                user_vector=user_vector,
                article_vector=article_vector,
                article_categories=article_categories,
                user_categories=user_categories,
                article_sentiment=article_sentiment,
                article_section=article_section,
                article_date=article_date
            )
            
            # Crear justificación
            justification = {
                'score': score,
                'matching_categories': list(set(user_categories) & set(article_categories)),
                'article_categories': article_categories,
                'sentiment': article_sentiment.get('label') if article_sentiment else None
            }
            
            scored_articles.append((article, score, justification))
        
        # Ordenar por score descendente
        scored_articles.sort(key=lambda x: x[1], reverse=True)
        
        # Retornar top_k
        return scored_articles[:top_k]

