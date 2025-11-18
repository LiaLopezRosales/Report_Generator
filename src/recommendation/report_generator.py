"""
Generador de reportes personalizados
"""
import logging
from typing import List, Dict
from datetime import datetime
from ..summarization.summarizer import PersonalizedSummarizer

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generador de reportes personalizados de noticias"""
    
    def __init__(self, summarizer: PersonalizedSummarizer):
        """
        Inicializa el generador de reportes
        
        Args:
            summarizer: Resumidor personalizado
        """
        self.summarizer = summarizer
    
    def generate_report(
        self,
        matched_articles: List[tuple],
        user_profile: Dict,
        max_articles: int = 10
    ) -> Dict:
        """
        Genera un reporte personalizado
        
        Args:
            matched_articles: Lista de tuplas (artículo, score, justificación)
            user_profile: Perfil del usuario
            max_articles: Número máximo de artículos en el reporte
            
        Returns:
            Diccionario con el reporte completo
        """
        user_categories = user_profile.get('categories', [])
        
        report_items = []
        
        for article, score, justification in matched_articles[:max_articles]:
            # Generar resumen personalizado
            article_text = article.get('text', '')
            summary = self.summarizer.summarize_for_profile(
                article_text,
                user_categories,
                num_sentences=3
            )
            
            report_item = {
                'article_id': article.get('id'),
                'title': article.get('title'),
                'url': article.get('url'),
                'section': article.get('section'),
                'summary': summary,
                'score': score,
                'justification': {
                    'matching_categories': justification.get('matching_categories', []),
                    'article_categories': justification.get('article_categories', []),
                    'sentiment': justification.get('sentiment'),
                    'score_breakdown': justification.get('score', 0)
                },
                'date': article.get('source_metadata', {}).get('date'),
                'tags': article.get('tags', [])
            }
            
            report_items.append(report_item)
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'user_profile': {
                'categories': user_categories,
                'profile_text': user_profile.get('profile_text', '')
            },
            'total_articles': len(matched_articles),
            'articles_in_report': len(report_items),
            'articles': report_items
        }
        
        return report
    
    def format_report_text(self, report: Dict) -> str:
        """
        Formatea el reporte como texto legible
        
        Args:
            report: Diccionario del reporte
            
        Returns:
            Texto formateado del reporte
        """
        lines = []
        lines.append("=" * 80)
        lines.append("REPORTE PERSONALIZADO DE NOTICIAS")
        lines.append("=" * 80)
        lines.append(f"\nGenerado: {report['generated_at']}")
        lines.append(f"Total de artículos relevantes: {report['total_articles']}")
        lines.append(f"Artículos en este reporte: {report['articles_in_report']}")
        lines.append("\n" + "-" * 80)
        
        for i, article in enumerate(report['articles'], 1):
            lines.append(f"\n{i}. {article['title']}")
            lines.append(f"   Sección: {article['section']}")
            lines.append(f"   Score de relevancia: {article['score']:.3f}")
            lines.append(f"\n   Resumen:")
            lines.append(f"   {article['summary']}")
            
            if article['justification']['matching_categories']:
                lines.append(f"\n   Categorías coincidentes: {', '.join(article['justification']['matching_categories'])}")
            
            if article['justification']['sentiment']:
                lines.append(f"   Sentimiento: {article['justification']['sentiment']}")
            
            lines.append(f"\n   URL: {article['url']}")
            lines.append("-" * 80)
        
        return "\n".join(lines)



