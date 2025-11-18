"""
Vectorización de noticias y perfiles de usuario
"""
import logging
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger(__name__)


class NewsVectorizer:
    """Vectorizador de noticias usando TF-IDF"""
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Inicializa el vectorizador
        
        Args:
            max_features: Número máximo de features
            ngram_range: Rango de n-gramas
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # Ya se eliminan en preprocesamiento
            lowercase=True
        )
        self.fitted = False
    
    def fit0(self, texts: List[str]):
        """
        Ajusta el vectorizador con textos
        
        Args:
            texts: Lista de textos preprocesados
        """
        self.vectorizer.fit(texts)
        self.fitted = True
    
    def transform0(self, texts: List[str]) -> np.ndarray:
        """
        Transforma textos a vectores
        
        Args:
            texts: Lista de textos preprocesados
            
        Returns:
            Matriz de vectores
        """
        if not self.fitted:
            raise ValueError("El vectorizador debe ser ajustado primero con fit()")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform0(self, texts: List[str]) -> np.ndarray:
        """
        Ajusta y transforma textos
        
        Args:
            texts: Lista de textos preprocesados
            
        Returns:
            Matriz de vectores
        """
        return self.vectorizer.fit_transform(texts).toarray()
    
    def vectorize_article(self, text: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Vectoriza un artículo individual
        
        Args:
            text: Texto del artículo
            metadata: Metadatos adicionales (sección, tags, categorías)
            
        Returns:
            Vector del artículo
        """
        vector = self.fit0([text])
        vector = self.transform0([text])[0]

        if not self.fitted:
            raise ValueError("El vectorizador debe ser ajustado primero con fit()")
        
        # Si hay metadatos, podrían agregarse como features adicionales
        # Por ahora, solo retornamos el vector TF-IDF
        
        return vector
    
    def get_feature_names(self) -> List[str]:
        """Obtiene los nombres de las features"""
        if not self.fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()


class UserProfileVectorizer:
    """Vectorizador de perfiles de usuario"""
    
    def __init__(self, news_vectorizer: NewsVectorizer):
        """
        Inicializa el vectorizador de perfiles
        
        Args:
            news_vectorizer: Vectorizador de noticias ya ajustado
        """
        self.news_vectorizer = news_vectorizer
    
    def vectorize_profile(self, profile_text: str, categories: Optional[List[str]] = None) -> np.ndarray:
        """
        Vectoriza un perfil de usuario
        
        Args:
            profile_text: Texto del perfil del usuario
            categories: Categorías de interés del usuario
            
        Returns:
            Vector del perfil
        """
        # Usar el mismo vectorizador que las noticias
        vector = self.news_vectorizer.vectorize_article(profile_text)
        
        # Las categorías podrían usarse para ponderar el vector
        # Por ahora, solo retornamos el vector basado en texto
        
        return vector



